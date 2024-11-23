import base64
import io
from typing import Any, Dict, List, Tuple

import PIL.Image
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_from_base64_gzipped_json

from ..logger import setup_logger

logger = setup_logger(__name__, "INFO")

# Constants
MAX_IMAGE_DIMENSION = 1000


def _process_image(base64_image: str) -> PIL.Image.Image:
    """Process and resize a base64 encoded image."""
    pil_image = PIL.Image.open(io.BytesIO(base64.b64decode(base64_image)))

    if (
        pil_image.size[0] > MAX_IMAGE_DIMENSION
        or pil_image.size[1] > MAX_IMAGE_DIMENSION
    ):
        ratio = min(
            MAX_IMAGE_DIMENSION / pil_image.size[0],
            MAX_IMAGE_DIMENSION / pil_image.size[1],
        )
        new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
        pil_image = pil_image.resize(new_size, PIL.Image.Resampling.LANCZOS)

    return pil_image


def multi_modal_extraction(
    file_path: str,
) -> Tuple[List[List[Any]], List[Dict[str, Any]]]:
    """
    Extract documents and images from a file.
    Args:
        file_path: Path to the file to process
    Returns:
        Tuple containing:
        - List of lists with text content and PIL images
        - List of metadata dictionaries
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be processed
    """
    logger.info(f"Extracting documents from {file_path}")

    try:
        elements = partition(
            filename=file_path,
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )

        chunks = chunk_by_title(elements)
        embedding_objects = []
        embedding_metadatas = []

        logger.info(f"Extracted {len(chunks)} chunks")

        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            metadata = chunk_dict["metadata"]

            embedding_object = [chunk_dict["text"]]
            metedata_dict = {
                "text": chunk_dict["text"],
                "filename": metadata["filename"],
                "page_number": metadata["page_number"],
                "last_modified": metadata["last_modified"],
                "languages": metadata["languages"],
                "filetype": metadata["filetype"],
            }

            # Process images if present
            if "orig_elements" in metadata:
                base64_elements_str = metadata["orig_elements"]
                eles = elements_from_base64_gzipped_json(base64_elements_str)
                image_data = []

                for ele in eles:
                    if ele.to_dict()["type"] == "Image":
                        base64_image = ele.to_dict()["metadata"]["image_base64"]
                        image_data.append(base64_image)
                        pil_image = _process_image(base64_image)
                        embedding_object.append(pil_image)
                        metedata_dict["image_data"] = image_data

            embedding_objects.append(embedding_object)
            embedding_metadatas.append(metedata_dict)

        return embedding_objects, embedding_metadatas

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error processing file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error extracting documents: {e}")
        raise
