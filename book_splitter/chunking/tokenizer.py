"""How a chunk length is measured, and what happens when a tokenizer is absent.

The chunk length is expressed in **tokens**, not characters, because the number
that matters downstream is how much of a translation prompt a chunk occupies.
The default is ``cl100k_base`` (the GPT-4 family encoding), which ``tiktoken``
can build offline once its vocabulary is cached.

Three things can go wrong, and none of them may stop a run:

* ``tiktoken`` is not installed;
* ``tiktoken`` is installed but its vocabulary cannot be fetched (offline,
  read-only cache);
* the caller asked for a Hugging Face tokenizer that chonkie cannot resolve.

In every case the stage falls back to :class:`ApproxTokenTokenizer`, a local,
deterministic approximation of about four characters per token, and records the
fallback in the manifest. A chunk is therefore never *unmeasured* — it is either
measured exactly or measured approximately, and the result says which.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

from ..core.constants import CHUNK_FALLBACK_TOKENIZER, DEFAULT_CHUNK_TOKENIZER

#: Tokenizers chonkie ships itself (no download, no optional package).
BUILTIN_TOKENIZERS: tuple[str, ...] = ("word", "character", "byte")
#: Encoding names ``tiktoken`` understands. ``gpt2`` is also a valid tiktoken
#: encoding name, not only a Hugging Face model.
TIKTOKEN_ENCODINGS: tuple[str, ...] = (
    "cl100k_base",
    "o200k_base",
    "p50k_base",
    "r50k_base",
    "gpt2",
)
#: Characters per token used by the local approximation. Four is the usual rule of
#: thumb for English; Serbian and other morphologically rich languages run a
#: little denser, so this is deliberately a conservative (slightly high) estimate.
APPROX_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class TokenizerChoice:
    """The tokenizer that was asked for and the one that will actually measure."""

    requested: str
    used: str
    tokenizer: Any = field(repr=False, compare=False)
    fallback: bool = False
    note: str = ""

    def count_tokens(self, text: str) -> int:
        """Count tokens with the resolved tokenizer."""
        counter = getattr(self.tokenizer, "count_tokens", None)
        if callable(counter):
            return int(counter(text))
        return len(self.tokenizer.encode(text))

    def to_settings(self) -> dict[str, Any]:
        """The part of this choice that belongs in a manifest."""
        return {
            "requested": self.requested,
            "used": self.used,
            "fallback": self.fallback,
            "note": self.note,
        }


def _approx_tokenizer_class():
    """Build the fallback class lazily, so ``chonkie`` stays an optional import."""
    from chonkie.tokenizer import Tokenizer

    class ApproxTokenTokenizer(Tokenizer):
        """A deterministic, offline stand-in for a real tokenizer.

        It measures about four characters per token and never touches the network.
        ``Tokenizer`` (chonkie's abstract base) only requires ``tokenize``,
        ``encode`` and ``decode``; the recursive chunker asks it for token counts
        at every level, which is all the chunker needs.
        """

        def __repr__(self) -> str:  # pragma: no cover - debugging aid
            return "ApproxTokenTokenizer(chars_per_token=4)"

        @staticmethod
        def _pieces(text: str) -> list[str]:
            return [
                text[i : i + APPROX_CHARS_PER_TOKEN]
                for i in range(0, len(text), APPROX_CHARS_PER_TOKEN)
            ]

        def tokenize(self, text: str) -> list[str]:
            return self._pieces(text)

        def encode(self, text: str) -> list[int]:
            encoded: list[int] = []
            for piece in self._pieces(text):
                token_id = self.token2id[piece]
                if token_id >= len(self.vocab):
                    self.vocab.append(piece)
                encoded.append(token_id)
            return encoded

        def decode(self, tokens) -> str:
            return "".join(self.vocab[token] for token in tokens)

        def count_tokens(self, text: str) -> int:
            return len(self._pieces(text)) if text else 0

    return ApproxTokenTokenizer()


def _wrap(tokenizer: Any):
    """Wrap a chonkie tokenizer (or a tokenizer name) in the common adapter."""
    from chonkie.tokenizer import AutoTokenizer

    return AutoTokenizer(tokenizer)


@functools.lru_cache(maxsize=8)
def resolve_tokenizer(name: str | None = None) -> TokenizerChoice:
    """Resolve ``name`` to a usable tokenizer, falling back to the local estimate.

    Resolution order is deliberate: an explicitly built-in tokenizer first, then
    ``tiktoken`` for a known encoding name, then chonkie's auto-loader (Hugging
    Face / tokie) for anything else. The same name resolves the same way for the
    life of the process, so a book is never measured by two different tokenizers
    between chapters.
    """
    requested = (name or DEFAULT_CHUNK_TOKENIZER).strip() or DEFAULT_CHUNK_TOKENIZER
    if requested in BUILTIN_TOKENIZERS:
        return TokenizerChoice(requested=requested, used=requested, tokenizer=_wrap(requested))

    if requested in TIKTOKEN_ENCODINGS:
        try:
            import tiktoken

            return TokenizerChoice(
                requested=requested,
                used=requested,
                tokenizer=_wrap(tiktoken.get_encoding(requested)),
            )
        except Exception as exc:  # noqa: BLE001 - any failure is a reason to fall back
            return _fallback(
                requested,
                f"tiktoken could not load {requested!r} ({type(exc).__name__}: {exc}); "
                f"install tiktoken (pip install tiktoken) for exact token counts",
            )

    try:
        return TokenizerChoice(requested=requested, used=requested, tokenizer=_wrap(requested))
    except Exception as exc:  # noqa: BLE001 - a missing HF model is not fatal
        return _fallback(
            requested,
            f"tokenizer {requested!r} could not be loaded ({type(exc).__name__}: {exc})",
        )


def _fallback(requested: str, reason: str) -> TokenizerChoice:
    """The local approximation, with the reason it was needed recorded."""
    tokenizer = _wrap(_approx_tokenizer_class())
    used = CHUNK_FALLBACK_TOKENIZER
    note = (
        f"{reason}; chunks are measured with the local ~{APPROX_CHARS_PER_TOKEN}-"
        f"characters-per-token estimate instead"
    )
    return TokenizerChoice(
        requested=requested,
        used=used,
        tokenizer=tokenizer,
        fallback=True,
        note=note,
    )


__all__ = [
    "APPROX_CHARS_PER_TOKEN",
    "BUILTIN_TOKENIZERS",
    "TIKTOKEN_ENCODINGS",
    "TokenizerChoice",
    "resolve_tokenizer",
]
