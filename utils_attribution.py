# utils_attribution.py
from __future__ import annotations

import hmac
import hashlib


def derive_message_bits(
    *,
    secret_key: str,
    api_key: str,
    user_id: str,
    request_time: str,
    nonce: str = "",
    bit_len: int = 16,
) -> str:
    """
    Deterministically generate a 0/1 bitstring from request metadata.
    AudioSeal supports 16 bits; use bit_len=16 for compatibility.
    Same inputs => same bits. Different user/time => different bits.
    """
    payload = f"api_key={api_key}|user_id={user_id}|request_time={request_time}|nonce={nonce}".encode("utf-8")
    digest = hmac.new(secret_key.encode("utf-8"), payload, hashlib.sha256).digest()  # 32 bytes

    bits = "".join(f"{b:08b}" for b in digest)  # 256 bits
    return bits[:bit_len]
