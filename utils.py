import base64
import json
import secrets
import zstandard


def dict_to_base64(dict_data: dict) -> str:
    return base64.b64encode(json.dumps(dict_data).encode("utf-8")).decode("utf-8")


def base64_to_dict(b64_data: str) -> dict:
    return json.loads(base64.b64decode(b64_data).decode("utf-8"))


def deflate_to_base64(filename: str) -> str:
    with open(filename, "rb") as f:
        data: bytes = f.read()
    compressor = zstandard.ZstdCompressor(level=10)
    compressed = compressor.compress(data)
    return base64.b64encode(compressed).decode("utf-8")


def inflate_to_file(data: str, filename: str) -> None:
    compressed: bytes = base64.b64decode(data)
    decompressor = zstandard.ZstdDecompressor()
    data: bytes = decompressor.decompress(compressed)
    with open(filename, "wb") as f:
        f.write(data)


def get_tmp_filename() -> str:
    return f"autosave{secrets.token_hex(4)}"


def fnv1a_64(s: str) -> int:
    fnv_prime = 0x100000001b3
    hash_val = 0xcbf29ce484222325

    for byte in s.encode("utf-8"):
        hash_val ^= byte
        hash_val = (hash_val * fnv_prime) % (1 << 64)
    return hash_val
