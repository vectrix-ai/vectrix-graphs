import base64

from pdf2image import convert_from_bytes
from PIL import Image


def pdf_to_image(
    pdf: str, return_base64: bool = False, max_dimensions: tuple[int, int] | None = None
) -> list:
    """
    Convert a base64 encoded PDF string to a list of PIL Image objects or base64 encoded images.
    Args:
        pdf (str): Base64 encoded string of a PDF file
        return_base64 (bool): If True, returns base64 encoded strings instead of PIL Images
        max_dimensions (tuple[int, int] | None): Maximum (width, height) for output images.
            Images will be scaled down maintaining aspect ratio if they exceed these dimensions.
    Returns:
        list: List of PIL Image objects or base64 encoded strings
    """
    try:
        # Decode base64 string to bytes
        pdf_bytes = base64.b64decode(pdf)

        # Convert PDF bytes to images
        images = convert_from_bytes(pdf_bytes)

        if max_dimensions:
            max_width, max_height = max_dimensions
            for i, img in enumerate(images):
                # Calculate scaling factor to maintain aspect ratio
                width, height = img.size
                scale = min(max_width / width, max_height / height)
                if scale < 1:
                    new_size = (int(width * scale), int(height * scale))
                    images[i] = img.resize(new_size, resample=Image.LANCZOS)

        if return_base64:
            # Convert PIL images to base64 JPG
            from io import BytesIO

            base64_images = []
            for img in images:
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                base64_images.append(base64.b64encode(buffered.getvalue()).decode())
            return base64_images

        return images

    except base64.binascii.Error:
        raise ValueError("Invalid base64 encoded string")
    except Exception as e:
        raise Exception(f"Error converting PDF to image: {str(e)}")
