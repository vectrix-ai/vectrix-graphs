import base64
import io
from datetime import datetime
from unittest.mock import Mock, mock_open, patch

import PIL.Image
import pytest

from vectrix_graphs.importers.documents import (
    _process_image,
    multi_modal_extraction,
)


# Test fixtures
@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    img = PIL.Image.new("RGB", (1500, 1500), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode()


@pytest.fixture
def mock_elements():
    """Create mock unstructured elements."""
    text_element = Mock()
    text_element.to_dict.return_value = {
        "text": "Sample text",
        "type": "Text",
        "metadata": {
            "filename": "test.pdf",
            "page_number": 1,
            "last_modified": datetime.now(),
            "languages": ["en"],
            "filetype": "pdf",
        },
    }

    image_element = Mock()
    image_element.to_dict.return_value = {
        "type": "Image",
        "metadata": {"image_base64": "mock_base64_string"},
    }

    return [text_element, image_element]


# Test cases
def test_process_image(sample_image):
    """Test image processing function."""
    # Process the sample image
    result = _process_image(sample_image)

    # Check if the result is a PIL Image
    assert isinstance(result, PIL.Image.Image)

    # Check if the image was resized correctly
    assert result.size[0] <= 1000
    assert result.size[1] <= 1000


def test_process_image_small():
    """Test processing of small images."""
    # Create a small image that shouldn't be resized
    img = PIL.Image.new("RGB", (500, 500), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    base64_img = base64.b64encode(img_byte_arr).decode()

    result = _process_image(base64_img)
    assert result.size == (500, 500)


@patch("vectrix_graphs.importers.documents.partition")
@patch("vectrix_graphs.importers.documents.chunk_by_title")
@patch("vectrix_graphs.importers.documents.elements_from_base64_gzipped_json")
def test_multi_modal_extraction(
    mock_elements_from_json, mock_chunk_by_title, mock_partition, mock_elements
):
    """Test the main document extraction function."""
    # Create a valid base64 image string
    img = PIL.Image.new("RGB", (100, 100), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    valid_base64_img = base64.b64encode(img_byte_arr).decode()

    # Setup mocks with valid image data
    image_element = Mock()
    image_element.to_dict.return_value = {
        "type": "Image",
        "metadata": {"image_base64": valid_base64_img},
    }
    text_element = Mock()
    text_element.to_dict.return_value = {
        "text": "Sample text",
        "type": "Text",
        "metadata": {
            "filename": "test.pdf",
            "page_number": 1,
            "last_modified": datetime.now(),
            "languages": ["en"],
            "filetype": "pdf",
        },
    }

    # Use the updated elements list
    mock_elements = [text_element, image_element]
    mock_elements_from_json.return_value = mock_elements
    mock_partition.return_value = mock_elements
    mock_chunk_by_title.return_value = [
        Mock(
            to_dict=lambda: {
                "text": "Sample text",
                "metadata": {
                    "filename": "test.pdf",
                    "page_number": 1,
                    "last_modified": datetime.now(),
                    "languages": ["en"],
                    "filetype": "application/pdf",
                    "orig_elements": "mock_base64_elements",
                },
            }
        )
    ]

    # Call the function
    embedding_objects, embedding_metadatas = multi_modal_extraction("test.pdf")

    # Assertions
    assert len(embedding_objects) == 1
    assert len(embedding_metadatas) == 1
    assert isinstance(embedding_objects[0], list)
    assert isinstance(embedding_metadatas[0], dict)
    assert "text" in embedding_metadatas[0]
    assert "filename" in embedding_metadatas[0]


def test_multi_modal_extraction_file_not_found():
    """Test handling of non-existent files."""
    with pytest.raises(FileNotFoundError):
        multi_modal_extraction("nonexistent.pdf")


@patch("vectrix_graphs.importers.documents.partition")
def test_multi_modal_extraction_value_error(mock_partition):
    """Test handling of processing errors."""
    mock_partition.side_effect = ValueError("Processing error")

    with pytest.raises(ValueError):
        multi_modal_extraction("test.pdf")


# Error cases
def test_process_image_invalid_base64():
    """Test handling of invalid base64 data."""
    with pytest.raises(Exception):
        _process_image("invalid_base64_string")


@pytest.mark.parametrize(
    "file_path",
    [
        "",
        None,
        "   ",
    ],
)
def test_multi_modal_extraction_invalid_path(file_path):
    """Test handling of invalid file paths."""
    with pytest.raises((ValueError, FileNotFoundError)):
        multi_modal_extraction(file_path)


@pytest.mark.integration
def test_multi_modal_extraction_with_mock_pdf():
    """Integration test using a mock PDF document."""
    test_pdf_path = "test_mock.pdf"

    # Mock all potential file operations
    with (
        patch(
            "vectrix_graphs.importers.documents.elements_from_base64_gzipped_json"
        ) as mock_elements,
        patch("builtins.open", mock_open(read_data=b"mock pdf content")),
        patch("os.path.exists") as mock_exists,
        patch("os.path.isfile") as mock_isfile,
        patch("pathlib.Path") as mock_path,
        patch("vectrix_graphs.importers.documents.partition") as mock_partition,
        patch(
            "vectrix_graphs.importers.documents.chunk_by_title"
        ) as mock_chunk_by_title,
    ):
        # Setup all file-related mocks
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = True
        mock_path.return_value = mock_path_instance

        # Setup mock elements
        text_element = Mock()
        text_element.to_dict.return_value = {
            "text": "Mock PDF content with some meaningful text",
            "type": "Text",
            "metadata": {
                "filename": test_pdf_path,
                "page_number": 1,
                "last_modified": datetime.now(),
                "languages": ["en"],
                "filetype": "application/pdf",
            },
        }

        image_element = Mock()
        # Create a small test image
        img = PIL.Image.new("RGB", (100, 100), color="red")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        valid_base64_img = base64.b64encode(img_byte_arr).decode()

        image_element.to_dict.return_value = {
            "type": "Image",
            "metadata": {"image_base64": valid_base64_img},
        }

        mock_elements.return_value = [text_element, image_element]
        mock_partition.return_value = [text_element, image_element]
        mock_chunk_by_title.return_value = [
            Mock(
                to_dict=lambda: {
                    "text": "Sample text",
                    "metadata": {
                        "filename": test_pdf_path,
                        "page_number": 1,
                        "last_modified": datetime.now(),
                        "languages": ["en"],
                        "filetype": "application/pdf",
                        "orig_elements": "mock_base64_elements",
                    },
                }
            )
        ]

        embedding_objects, embedding_metadatas = multi_modal_extraction(test_pdf_path)

        # Basic validation
        assert len(embedding_objects) > 0
        assert len(embedding_objects) == len(embedding_metadatas)

        # Check metadata structure
        for metadata in embedding_metadatas:
            assert "text" in metadata
            assert "filename" in metadata
            assert metadata["filename"] == test_pdf_path
            assert "page_number" in metadata
            assert "filetype" in metadata
            assert metadata["filetype"] == "application/pdf"
