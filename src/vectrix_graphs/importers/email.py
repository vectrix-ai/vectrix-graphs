from langchain_core.documents import Document
from O365 import Account

from ..logger import setup_logger

logger = setup_logger(__name__, "INFO")


class EmailImporter:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        email_address: str,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.email_address = email_address

        self.account = Account(
            (self.client_id, self.client_secret),
            tenant_id=self.tenant_id,
            auth_flow_type="credentials",
            main_resource=self.email_address,
        )

        if not self.account.authenticate():
            logger.error("Failed to authenticate")
            raise Exception("Failed to authenticate")

        self.mailbox = self.account.mailbox()

    def _download_attachments(self, attachments, allow_types: list[str] = []) -> dict:
        attachments_ids = []
        for attachment in attachments:
            # Skip if allow_types is specified and attachment type doesn't match
            if allow_types and not any(
                attachment.name.lower().endswith(t.lower()) for t in allow_types
            ):
                logger.debug(
                    f"Skipping attachment {attachment.name} - type not allowed"
                )
                continue

            # Get the raw bytes using the download() method

            attachments_ids.append(
                {
                    "size": attachment.size,
                    "name": attachment.name,
                    "content_bytes": attachment.content,
                }
            )
        return attachments_ids

    def _extract_metadata(self, message) -> dict:
        return {
            "object_id": message.object_id,
            "subject": message.subject,
            "sender": message.sender,
            "to": [{"address": to.address, "name": to.name} for to in message.to],
            "cc": [{"address": cc.address, "name": cc.name} for cc in message.cc],
            "bcc": [{"address": bcc.address, "name": bcc.name} for bcc in message.bcc],
            "reply_to": [
                {"address": reply_to.address, "name": reply_to.name}
                for reply_to in message.reply_to
            ],
            "created": message.created,
            "received": message.received,
            "modified": message.modified,
            "sent": message.sent,
            "categories": message.categories,
            "importance": message.importance,
            "is_read": message.is_read,
            "is_draft": message.is_draft,
            "flag": message.flag,
        }

    def get_messages(
        self,
        filter_attachment_type: list[str] = [],
        folder_name: str = "inbox",
        limit: int = 10,
        download_attachments: bool = False,
    ) -> list[Document]:
        """Retrieve messages from a specified folder in the mailbox.

        Args:
            filter_attachment_type (list[str], optional): List of allowed attachment file extensions. Defaults to [].
            folder_name (str, optional): Name of the folder to retrieve messages from. Defaults to "inbox".
            limit (int, optional): Maximum number of messages to retrieve. Defaults to 10.
            download_attachments (bool, optional): Whether to download message attachments. Defaults to False.

        Returns:
            list[Document]: List of Document objects containing:
                - page_content: Message body text
                - metadata: Dict with message metadata including:
                    - Email headers (subject, sender, recipients, etc)
                    - Timestamps (created, received, modified, sent)
                    - Message flags and categories
                    - attachments: List of dicts with attachment info:
                        - size: Attachment size in bytes
                        - name: Filename
                        - content_bytes: Base64 encoded attachment content
        """
        folder = self.mailbox.get_folder(folder_name=folder_name)
        results = folder.get_messages(limit=limit, download_attachments=True)

        messages = []
        for result in results:
            attachments_ids = []
            if result.has_attachments and download_attachments:
                # Download attachments
                attachments = list(result.attachments)
                attachments_ids = self._download_attachments(
                    attachments, filter_attachment_type
                )
                logger.info(f"Downloaded {len(attachments_ids)} attachments")

            logger.debug(str(result.subject))

            # Create message
            page_content = (result.get_body_text(),)
            metadata = self._extract_metadata(result)
            metadata["attachments"] = attachments_ids

            messages.append(Document(page_content=str(page_content), metadata=metadata))
        return messages

    def get_message_by_id(
        self, message_id: str, filter_attachment_type: list[str] = []
    ) -> Document:
        """Retrieve a specific email message by its ID.

        Args:
            message_id (str): The unique identifier of the email message to retrieve.
            filter_attachment_type (list[str], optional): List of allowed attachment file extensions. Defaults to [].

        Returns:
            Document: A Document object containing:
                - page_content: Message body text
                - metadata: Dict with message metadata including:
                    - Email headers (subject, sender, recipients, etc)
                    - Timestamps (created, received, modified, sent)
                    - Message flags and categories
                    - attachments: List of attachment IDs
        """
        message = self.mailbox.get_message(message_id, download_attachments=True)
        attachments_ids = self._download_attachments(
            message.attachments, filter_attachment_type
        )
        metadata = self._extract_metadata(message)
        metadata["attachments"] = attachments_ids
        return Document(page_content=str(message.get_body_text()), metadata=metadata)
