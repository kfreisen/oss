#!/usr/bin/env python3
"""Fail if any package pyproject.toml defines its own ruff configuration.

Ruff treats a ``pyproject.toml`` as a configuration file only when it contains a
``[tool.ruff]`` table. The moment a package adds one, ruff stops walking up to the
repository-root ``ruff.toml`` for every file under that package — silently, with no
error and no diff in the lint output until the two configs disagree.

There is one ruff config in this repo and it lives at the root. This hook keeps it
that way.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matches a [tool.ruff] table header or any sub-table of it. Deliberately a regex
# rather than tomllib: this runs as a pre-commit `language: system` hook against
# whatever python3 the contributor has, and tomllib is 3.11+.
RUFF_TABLE = re.compile(r"^\s*\[\s*tool\.ruff(\.|\s*\])", re.MULTILINE)


def has_ruff_table(path: Path) -> bool:
    """Return whether ``path`` contains a ``[tool.ruff]`` table."""
    return RUFF_TABLE.search(path.read_text(encoding="utf-8")) is not None


def main(argv: list[str]) -> int:
    """Check the given pyproject files, or every package pyproject if none are given."""
    paths = [Path(a) for a in argv] or sorted(REPO_ROOT.glob("packages/*/pyproject.toml"))
    offenders = [p for p in paths if p.name == "pyproject.toml" and has_ruff_table(p)]

    for path in offenders:
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        print(
            f"{rel}: defines [tool.ruff]. Ruff would use this file as the config root "
            f"and ignore the repository-root ruff.toml for this package. "
            f"Remove the table; configure ruff in /ruff.toml instead.",
            file=sys.stderr,
        )
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
