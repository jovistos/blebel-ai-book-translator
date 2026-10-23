"""Picture embedding for the EPUB package."""

from __future__ import annotations

import posixpath
from pathlib import Path

from ..core.artifacts import sha256_bytes
from ..core.constants import (
    EPUB_IMAGES_DIRNAME,
    IMAGES_DIRNAME,
    MARKDOWN_DIRNAME,
)
from ..core.errors import ToolError
from .package import ImageAsset, media_type_for
from .render import HTML_IMG_SRC_RE, rewrite_images

#: Self-contained pictures never travel into the package.
_REMOTE_PREFIXES = ("http://", "https://", "data:", "//")


def embed_images(
    body: str,
    *,
    base_dir: Path,
    assets: dict[str, ImageAsset],
    used_names: set[str],
    name: str,
    errors: list[str],
    warnings: list[str],
) -> str:
    """Copy every referenced picture into the container and rewrite its ``src``."""

    def handler(src: str, tag: str) -> str:
        if not src:
            return ""
        if src.lower().startswith(_REMOTE_PREFIXES):
            warnings.append(f"{name}: remote picture left as a link: {src}")
            return tag
        path = Path(src) if Path(src).is_absolute() else base_dir / src
        key = str(path.resolve()) if path.exists() else str(path)
        asset = assets.get(key)
        if asset is None:
            if not path.is_file():
                errors.append(
                    f"{name}: picture not found: {src} "
                    "(run `convert` with images, or restore 00-markdown/images/)"
                )
                return f'<span class="image-missing">[{escape(src)}]</span>'
            data = path.read_bytes()
            filename = unique_name(path.name, used_names)
            asset = ImageAsset(
                href=f"{EPUB_IMAGES_DIRNAME}/{filename}",
                media_type=media_type_for(filename),
                data=data,
                source=str(path),
                sha256=sha256_bytes(data),
            )
            if asset.media_type == "application/octet-stream":
                warnings.append(f"{name}: unknown picture type, declared as bytes: {src}")
            assets[key] = asset
        return replace_src(tag, f"../{asset.href}")

    body, _ = rewrite_images(body, handler)
    return body


def cover_asset(
    cover: str | Path,
    source: Path,
    assets: dict[str, ImageAsset],
    used_names: set[str],
    warnings: list[str],
) -> ImageAsset:
    """Resolve ``--cover`` against the book folder and register it once."""
    path = resolve_cover(cover, source)
    key = str(path.resolve())
    asset = assets.get(key)
    if asset is not None:
        return asset
    data = path.read_bytes()
    filename = unique_name(path.name, used_names)
    asset = ImageAsset(
        href=f"{EPUB_IMAGES_DIRNAME}/{filename}",
        media_type=media_type_for(filename),
        data=data,
        source=str(path),
        sha256=sha256_bytes(data),
    )
    if asset.media_type == "application/octet-stream":
        warnings.append(f"cover picture type is unknown, declared as bytes: {path}")
    assets[key] = asset
    return asset


def resolve_cover(cover: str | Path, source: Path) -> Path:
    """A cover path, or the same name inside the book's extracted pictures."""
    candidate = Path(cover).expanduser()
    if candidate.is_file():
        return candidate
    pictures = source.parent / MARKDOWN_DIRNAME / IMAGES_DIRNAME
    for option in (pictures / candidate.name, pictures / str(cover)):
        if option.is_file():
            return option
    raise ToolError(
        f"cover picture not found: {cover} (looked next to it and in {pictures})"
    )


def unique_name(name: str, used: set[str]) -> str:
    """A safe, unique file name inside ``images/``."""
    safe = "".join(
        character
        if character.isalnum() or character in {".", "-", "_"}
        else "-"
        for character in Path(name).name
    ).strip("-") or "image"
    stem, suffix = posixpath.splitext(safe)
    candidate = safe
    bump = 2
    while candidate in used:
        candidate = f"{stem}-{bump}{suffix}"
        bump += 1
    used.add(candidate)
    return candidate


def replace_src(tag: str, new_src: str) -> str:
    """Replace the ``src`` attribute of an ``<img>`` tag, keeping the rest."""
    return HTML_IMG_SRC_RE.sub(lambda _match: f'src="{new_src}"', tag, count=1)


def escape(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


__all__ = [
    "cover_asset",
    "embed_images",
    "escape",
    "replace_src",
    "resolve_cover",
    "unique_name",
]
