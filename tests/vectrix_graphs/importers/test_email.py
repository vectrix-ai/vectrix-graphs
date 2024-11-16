from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from vectrix_graphs.importers import EmailImporter


@pytest.fixture
def mock_account():
    with patch("vectrix_graphs.importers.email.Account") as mock:
        account = Mock()
        account.authenticate.return_value = True
        mock.return_value = account
        yield mock


@pytest.fixture
def email_importer(mock_account):
    return EmailImporter(
        client_id="fake_id",
        client_secret="fake_secret",
        tenant_id="fake_tenant",
        email_address="test@example.com",
    )


def test_init_authentication_failure(mock_account):
    mock_account.return_value.authenticate.return_value = False

    with pytest.raises(Exception, match="Failed to authenticate"):
        EmailImporter(
            client_id="fake_id",
            client_secret="fake_secret",
            tenant_id="fake_tenant",
            email_address="test@example.com",
        )


def test_get_messages(email_importer):
    # Mock message data
    mock_message = Mock()
    mock_message.subject = "Test Subject"
    mock_message.sender = "sender@example.com"
    mock_message.to = [Mock(address="to@example.com", name="To Person")]
    mock_message.cc = []
    mock_message.bcc = []
    mock_message.reply_to = []
    mock_message.created = datetime.now()
    mock_message.received = datetime.now()
    mock_message.modified = datetime.now()
    mock_message.sent = datetime.now()
    mock_message.categories = []
    mock_message.importance = "normal"
    mock_message.is_read = True
    mock_message.is_draft = False
    mock_message.flag = None
    mock_message.has_attachments = False
    mock_message.get_body_text.return_value = "Test body"
    mock_message.object_id = "123"

    # Mock folder and results
    mock_folder = Mock()
    mock_folder.get_messages.return_value = [mock_message]
    email_importer.mailbox.get_folder.return_value = mock_folder

    # Test get_messages
    messages = email_importer.get_messages(folder_name="inbox", limit=1)

    assert len(messages) == 1
    assert isinstance(messages[0], Document)
    assert messages[0].metadata["subject"] == "Test Subject"
    assert messages[0].metadata["sender"] == "sender@example.com"
    assert len(messages[0].metadata["attachments"]) == 0


def test_download_attachments(email_importer):
    # Mock attachment
    mock_attachment = Mock()
    mock_attachment.name = "test.pdf"
    mock_attachment.size = 1024
    mock_attachment.content = b"fake content"

    result = email_importer._download_attachments(
        [mock_attachment], allow_types=[".pdf"]
    )

    assert len(result) == 1
    assert result[0]["name"] == "test.pdf"
    assert result[0]["size"] == 1024
    assert result[0]["content_bytes"] == b"fake content"


def test_download_attachments_filtered(email_importer):
    # Mock attachment with non-matching type
    mock_attachment = Mock()
    mock_attachment.name = "test.jpg"
    mock_attachment.size = 1024

    result = email_importer._download_attachments(
        [mock_attachment], allow_types=[".pdf"]
    )

    assert len(result) == 0
