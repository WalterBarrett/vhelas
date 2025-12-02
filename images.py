import os

from PIL import Image

mime_map = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "WEBP": "image/webp",
}


def get_mime_type(file_path: str):
    if not os.path.exists(file_path):
        return None
    try:
        with Image.open(file_path) as img:
            return mime_map.get(img.format, "application/octet-stream")
    except Exception:
        return None
