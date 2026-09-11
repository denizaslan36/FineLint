from __future__ import annotations

import re
import unicodedata
import zlib
from collections.abc import Iterable

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Normalize superficial differences without changing punctuation or wording."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def character_shingles(text: str, size: int = 5) -> set[str]:
    if len(text) <= size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


_MASK_64 = (1 << 64) - 1
_MINHASH_PARAMETERS = (
    (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F),
    (0xD6E8FEB86659FD93, 0xA5A3564E27F8862D),
    (0x94D049BB133111EB, 0x369DEA0F31A53F85),
    (0xBF58476D1CE4E5B9, 0xDB4F0B9175AE2165),
    (0xA24BAED4963EE407, 0x9FB21C651E98DF25),
    (0x9E6C63D0676A9A99, 0xC6BC279692B5CC83),
    (0xF1357AEA2E62A9C5, 0xB492B66FBE98F273),
    (0xD1342543DE82EF95, 0x8CB92BA72F3D8DD7),
)


def minhash_signature(features: Iterable[str]) -> tuple[int, ...]:
    """Return a compact deterministic MinHash signature for LSH candidates."""
    minima = [_MASK_64] * len(_MINHASH_PARAMETERS)
    found = False
    for feature in features:
        found = True
        encoded = feature.encode("utf-8", errors="surrogatepass")
        low = zlib.crc32(encoded)
        high = zlib.crc32(encoded, 0xA5A5A5A5)
        value = (high << 32) | low
        for index, (multiplier, offset) in enumerate(_MINHASH_PARAMETERS):
            permuted = (multiplier * value + offset) & _MASK_64
            minima[index] = min(minima[index], permuted)
    return tuple(minima) if found else ()


def minhash_bands(signature: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    # One value per band maximizes recall; Jaccard verification removes false matches.
    return tuple(enumerate(signature))
