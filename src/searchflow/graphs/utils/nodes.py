from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticToolsParser, JsonOutputParser, StrOutputParser
from langgraph.constants import Send


from .state import OverallState, Intent, QuestionList, QuestionState, CitedSources

class GraphNodes:
    def __init__(self, logger, vector_db):
        self.logger = logger
        self.logger.info("GraphNodes initialized")
        self.vectordb = vector_db

    @staticmethod
    def _setup_intent_detection():
        prompt = hub.pull("vectrix/intent_detection")
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        llm_with_tools = llm.bind_tools(tools=[Intent])
        return prompt | llm_with_tools
    
    @staticmethod
    def _setup_question_detection():
        prompt = hub.pull("vectrix/split_questions")
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        llm_with_tools = llm.bind_tools(tools=[QuestionList])
        return prompt | llm_with_tools
    
    @staticmethod
    def _rag_answer_chain():
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        prompt_template = hub.pull("answer_question")
        return prompt_template | llm | StrOutputParser()
    
    @staticmethod
    def _setup_cite_sources_chain():
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_with_tools = llm.bind_tools([CitedSources])
        prompt = hub.pull("cite_sources")
        return prompt | llm_with_tools | PydanticToolsParser(tools=[CitedSources])
    
    @staticmethod
    def _question_rewriter_chain():
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = hub.pull("vectrix/question_rewriter")
        return prompt | llm | StrOutputParser()
    

    @staticmethod
    def _setup_hallucination_grader():
        prompt = hub.pull("vectrix/hallucination_prompt")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return prompt | llm


    async def detect_intent(self, state :OverallState, config):
        self.logger.info("Detecting intent")
        messages = state["messages"]
        question = messages[-1].content
        # Select all messages except the last one
        chat_history = messages[:-1]
        intent_detection = self._setup_intent_detection()
        response = await intent_detection.ainvoke({"chat_history": chat_history, "question": question})
        self.logger.info(f"Intent detection response: {response['intent']}")
        return {"intent": response['intent']}
    

    async def decide_answering_path(self,state :OverallState, config):
        self.logger.info(f"Deciding answering path for intent: {state['intent']}")
        if state["intent"] == "greeting":
            return "greeting"
        elif state["intent"] == "specific_question":
            return "specific_question"
        elif state["intent"] == "metadata_query":
            return "metadata_query"
        elif state["intent"] == "follow_up_question":
            return "follow_up_question"
        else:
            return "end"
        
    async def split_question_list(self,state: OverallState, config):
        self.logger.info("Splitting question list")
        split_questions = self._setup_question_detection()
        question = state['messages'][-1].content
        questions = await split_questions.ainvoke({"QUESTION": question})
        self.logger.info("Question was split into %s parts", len(questions))
        return {"question_list": questions}
    
    async def llm_answer(self, state :OverallState, config):
        self.logger.info("Answering question with LLM")
        messages = state["messages"]
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        response = await llm.ainvoke(messages)
        response = AIMessage(content=response.content)
        return {"messages": response}
    
    async def retrieve_documents(self, state: OverallState, config):
        """
        We will perform a vector search for all question and return the top documents for eacht question
        """
        # Initiate the documents list
        self.logger.info("Retrieving documents, for the following questions: %s", state["question_list"]['questions'])
        return [Send("retrieve", {"question": q}) for q in state["question_list"]['questions']]
    

    async def retrieve(self, state: QuestionState, config):
        '''
        Retrieve documents relevant to the question

        Args:
            state: GraphState

        Returns:
            state (dict): Updates documents key with relevant documents
        '''
        self.logger.info("Retrieving documents")
        question = state['question']
        results = self.vectordb.similarity_search(
            query=question,
            k=3
        )
        # Filter all documents with a cosine distance smaller than 0.45
        filtered_documents = [doc for doc in results if doc.metadata['cosine_distance'] < 0.45]

        return {"documents": filtered_documents}
    


    async def rag_answer(self, state: OverallState, config):
        self.logger.info(f"Answering question based on {len(state['documents'])} retrieved documents")
        question = state["messages"][-1].content
        
        sources = ""
        for i, doc in enumerate(state["documents"], 1):
            sources += f"{i}. {doc.page_content}\n\n"

        final_answer_chain = self._rag_answer_chain()

        response = await final_answer_chain.ainvoke({"SOURCES": sources, "QUESTION": question})
        response = AIMessage(content=response)

        return {"temporary_answer": response}
    
    async def hallucination_grader(self, state: OverallState, config):
        self.logger.info("Grading hallucination")
        answer = state["temporary_answer"]
        documents = state["documents"]
        hallucination_grader = self._setup_hallucination_grader()
        response = await hallucination_grader.ainvoke({"documents": documents, "generation": answer})
        grade = response["binary_score"]
        return {"hallucination_grade": grade}
    

    
    async def grade(self, state: OverallState, config):
        if state["hallucination_grade"]:
            self.logger.info("No hallucinations detected")
            return "no_hallucinations"
        else:
            self.logger.info("Hallucinations detected")
            return "hallucinations"
        

    async def rewrite_question(self, state: OverallState, config):
        question_rewriter = self._setup_question_rewriter_chain()
        question = state["messages"][-1].content
        rewritten_question = await question_rewriter.ainvoke({"question": question})
        return {"messages": rewritten_question}


    async def cite_sources(self, state: OverallState, config):
        question = state["messages"][-1].content

        sources = ""

        if len(state["documents"]) == 0:
            self.logger.error('Unable to answer, no sources found')
            return {"cited_sources": ""}
        
        for i, doc in enumerate(state["documents"], 1):
            source = doc.metadata.get('source', 'Unknown')
            url = doc.metadata.get('url', 'No URL provided')
            sources += f"{i}. {doc.page_content}\n\nURL: {url}\nSOURCE: {source}\n"

        cite_sources_chain = self._setup_cite_sources_chain()

        response = await cite_sources_chain.ainvoke({"SOURCES": sources, "QUESTION": question})

        return {"cited_sources": response}
    
    async def final_answer(self,state: OverallState, config):
        final_answer = state["temporary_answer"]
        return {"messages": final_answer}
    




        


