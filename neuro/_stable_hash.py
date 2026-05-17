from __future__ import annotations
import hashlib

def stable_hash(sid, mod: int=1000) -> int:
    return int.from_bytes(hashlib.md5(str(sid).encode('utf-8')).digest()[:4], 'big') % mod
