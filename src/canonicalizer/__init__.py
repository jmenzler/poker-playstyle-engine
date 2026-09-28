"""src/canonicalizer — public interface for GameState embedding.

Public exports:
    Canonicalizer  — encode(GameState) → EncodeResult
    EncodeResult   — NamedTuple(embedding, hard_filter, schema_version)

Phase 3 consumers import from this package:
    from src.canonicalizer import Canonicalizer, EncodeResult
"""

from src.canonicalizer.encoder import Canonicalizer, EncodeResult

__all__ = ["Canonicalizer", "EncodeResult"]
