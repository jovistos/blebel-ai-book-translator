# external/ — NOT part of this repository

Everything in this folder is **third-party code that does not belong to this project**.
It is excluded from version control (see `.gitignore`): nothing inside `external/` is ever
committed from here, and the only tracked file is this README.

This folder exists purely as a local, clearly-marked landing area, so a checkout of an
upstream project can sit next to the book translator without ever being mistaken for part of it.

## Contents

| Folder | What it is | Its own repository |
| --- | --- | --- |
| `TranslateBooksWithLLMs/` | Upstream translation tool (`https://github.com/hydropix/TranslateBooksWithLLMs.git`, checked out at `v1.5.10`), vendored for local side-by-side use | yes — it has its own `.git` and history; commit and pull there directly |

## Rules

1. **Never commit anything from `external/` into this repository.** Only this README is
   tracked; the rest of the folder is gitignored.
2. **Never edit vendored code here.** If a change matters, make it in the upstream project
   (inside its own repository) and pull it back down.
3. **This project must never import from `external/`.** `book_translator` is self-contained; its
   dependencies are declared in `requirements-book-translator.txt`, not borrowed from a vendored
   copy.

## Restoring the folder on a fresh clone

`external/` is empty after a clone (only this README is present). Re-create the checkout with:

```bash
mkdir -p external
git clone https://github.com/hydropix/TranslateBooksWithLLMs external/TranslateBooksWithLLMs
```

The book translator works without it — the folder is a convenience for local work only.
