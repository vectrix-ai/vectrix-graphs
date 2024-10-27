from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain import hub
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticToolsParser, StrOutputParser, XMLOutputParser
from langgraph.constants import Send
from typing import List

from .state import OverallState, Intent, QuestionList, QuestionState, CitedSources

class GraphNodes:
    def __init__(self, logger, vector_db, mode='local'):
        if mode not in ['local', 'online']:
            raise ValueError("Mode must be either 'local' or 'online'")
        self.logger = logger
        self.logger.info("GraphNodes initialized")
        self.vectordb = vector_db
        self.mode = mode

    @staticmethod
    def _setup_intent_detection(mode):
        prompt = hub.pull("vectrix/intent_detection")
        if mode == 'online':
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        else:
            llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", temperature=0)
        llm_with_tools = llm.bind_tools(tools=[Intent])
        return prompt | llm_with_tools
    
    @staticmethod
    def _setup_question_detection(mode):
        prompt = hub.pull("vectrix/split_questions")
        if mode == 'online':
            llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        else:
            llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", temperature=0)
        llm_with_tools = llm.bind_tools(tools=[QuestionList])
        return prompt | llm_with_tools
    
    @staticmethod
    def _rag_answer_chain(mode):
        if mode == 'online':
            prompt_template = hub.pull("answer_question")
            llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022", temperature=0, max_tokens=6000)
            return prompt_template | llm
        else:
            prompt_template = hub.pull("vectrix/answer_question_local")
            llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", temperature=0, max_tokens=6000)
            return prompt_template | llm | XMLOutputParser(tags=["answer"])
        
    
    @staticmethod
    def _setup_cite_sources_chain(mode):
        if mode == 'online':
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", temperature=0)
        llm_with_tools = llm.bind_tools([CitedSources])
        prompt = hub.pull("cite_sources")
        return prompt | llm_with_tools | PydanticToolsParser(tools=[CitedSources])
    
    @staticmethod
    def _question_rewriter_chain(mode):
        if mode == 'online':
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", temperature=0)
        prompt = hub.pull("vectrix/question_rewriter")
        return prompt | llm | StrOutputParser()
    

    @staticmethod
    def _setup_hallucination_grader(mode):
        if mode == 'online':
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", temperature=0)
        prompt = hub.pull("vectrix/hallucination_prompt")
        return prompt | llm
    
    @staticmethod
    def _filter_duplicate_docs(documents : List[Document]) -> List[Document]:
        seen_uuids = set()
        unique_documents = []
        for doc in documents:
            uuid = doc.metadata.get('uuid')
            if uuid not in seen_uuids:
                seen_uuids.add(uuid)
                unique_documents.append(doc)
        return unique_documents
    
    @staticmethod
    def _rewrite_chat_history(mode, messages):
        prompt = hub.pull("question_context_reformulation")

        if mode == 'online':
            llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=6000)
        else:
            llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=6000)
        
        return prompt | llm

    async def detect_message_history(self, state: OverallState, config):
        if len(state["messages"]) > 1:
            return "True"
        else:
            return "False"

    async def rewrite_chat_history(self, state: OverallState, config):
        rewritten_question = self._rewrite_chat_history(self.mode, state["messages"])

        question = state["messages"][-1].content
        chat_history = ""
        for message in state["messages"][:-1]:
            chat_history += f"{message.type}: {message.content}\n"
        result = await rewritten_question.ainvoke({"USER_QUESTION": question, "CHAT_HISTORY": chat_history})
        return {"messages": result["reformulated_question"]}
    

    async def detect_intent(self, state :OverallState, config):
        self.logger.info("Detecting intent")
        messages = state["messages"]
        question = messages[-1].content
        # Select all messages except the last one
        chat_history = messages[:-1]
        intent_detection = self._setup_intent_detection(self.mode)
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
        split_questions = self._setup_question_detection(self.mode)
        question = state['messages'][-1].content
        questions = await split_questions.ainvoke({"QUESTION": question})
        self.logger.info("Question was split into %s parts", len(questions))
        return {"question_list": questions}
    
    async def llm_answer(self, state :OverallState, config):
        self.logger.info("Answering question with LLM")
        messages = state["messages"]
        if self.mode == 'online':
            llm = ChatOpenAI(temperature=0, model="gpt-4o")
        else:
            llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", temperature=0)
        response = await llm.ainvoke(messages)
        response = AIMessage(content=response.content)
        return {"messages": response}
    
    async def retrieve_documents(self, state: OverallState, config):
        """
        We will perform a vector search for all question and return the top documents for eacht question
        """
        # Initiate the documents list
        self.logger.info("Answering the following questions: %s", state["question_list"]['questions'])
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
        #filtered_documents = [doc for doc in results if doc.metadata['cosine_distance'] < 0.8]

        return {"documents": results}
    
    async def filter_docs(self, state: OverallState, config):
        documents = state["documents"]
        if len(documents) == 0:
            return {"documents": []}
        
        filtered_docs = self._filter_duplicate_docs(documents)

        state['documents'].clear()
        return {"documents": filtered_docs}
    


    async def rag_answer(self, state: OverallState, config):
        self.logger.info(f"Answering question based on {len(state['documents'])} retrieved documents")
        question = state["messages"][-1].content
        
        sources = ""
        for i, doc in enumerate(state["documents"], 1):
            sources += f"{i}. {doc.page_content}\n\n"

        final_answer_chain = self._rag_answer_chain(self.mode)
        response = await final_answer_chain.ainvoke({"SOURCES": sources, "QUESTION": question})

        if self.mode == 'online':
            # Extract the content from the 'answer_markdown' key
            answer_content = response['answer']
        else:
            answer_content = response['answer']

        # Create the AIMessage with the extracted content
        ai_message = AIMessage(content=answer_content)

        return {"temporary_answer": ai_message}
    
    async def final_answer(self, state: OverallState, config):
        self.logger.info("Final answer: %s", state["temporary_answer"])
        return {"messages": state["temporary_answer"]}
    
    async def hallucination_grader(self, state: OverallState, config):
        self.logger.info("Grading hallucination")
        answer = state["temporary_answer"]
        documents = state["documents"]
        hallucination_grader = self._setup_hallucination_grader(self.mode)
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
        question_rewriter = self._setup_question_rewriter_chain(self.mode)
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

        cite_sources_chain = self._setup_cite_sources_chain(self.mode)

        response = await cite_sources_chain.ainvoke({"SOURCES": sources, "QUESTION": question})

        return {"cited_sources": response}
    
    async def metadata_query(self, state: OverallState, config):
        self.logger.info("Answering metadata query")
        return {"messages": [AIMessage(content="A metatadata query is currently not supported in this demo. Please contact Ben Selleslagh (ben@vectrix.ai) for more information.")]}




        



