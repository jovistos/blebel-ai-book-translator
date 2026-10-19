"""The shape of the model's answer: one translated chunk.

Stage 7 is a *translation* call, but it is still asked for a JSON object rather
than free text. That is deliberate:

* the answer can be validated with Pydantic before it is allowed near the book,
  so a refusal or an empty field is a retry instead of a hole in a chapter;
* native structured output is available on most providers, which removes the
  "Here is the translation:" preamble and the code fence without prompt-hacking;
* an object-rooted schema is what every provider's structured-output mode
  accepts, and it leaves room for a second field later without breaking the run.

``ChunkTranslationAnswer`` has exactly one field. A translation is the whole
answer, not a summary of it, so there is nowhere else for the model to put
commentary — and ``extra="ignore"`` means a model that adds a helpful ``notes``
field anyway still produces a usable translation instead of a failed chapter.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChunkTranslationAnswer(BaseModel):
    """One chunk's translated text, as the model returns it.

    The validator rejects whitespace-only answers, because an empty string that
    validates is worse than a failed call: it would be written as an empty chunk
    and only noticed when the reconstructed chapter lost a paragraph.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    translation: str = Field(
        ...,
        min_length=1,
        description=(
            "The complete translation of the source chunk, with its Markdown "
            "structure and paragraph breaks preserved. No commentary."
        ),
    )

    @field_validator("translation")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not str(value or "").strip():
            raise ValueError("the translation is empty")
        return value

    @property
    def text(self) -> str:
        return self.translation


__all__ = ["ChunkTranslationAnswer"]
