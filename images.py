import io
import os
from base64 import b64encode
from io import BytesIO

from PIL import Image, PngImagePlugin

mime_map = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "WEBP": "image/webp",
}


def get_mime_type(file_path: str) -> str | None:
    if not os.path.exists(file_path):
        return None
    try:
        with Image.open(file_path) as img:
            return mime_map.get(img.format, "application/octet-stream")
    except Exception:
        return None


def merge_image_and_json(image_path: str, json: str) -> BytesIO | None:
    if not os.path.exists(image_path):
        return None
    img = Image.open(image_path).convert("RGBA")  # normalize
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("chara", b64encode(json.encode('utf-8')).decode('utf-8'))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", pnginfo=pnginfo)
    buffer.seek(0)
    return buffer
