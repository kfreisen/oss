#!/usr/bin/env python3
"""Fail if one package in this repo imports another.

The five packages publish independently to PyPI. An import from one into another
would either be a broken distribution (the import is not a declared dependency, so
it works in the monorepo and fails for users) or a hidden coupling that forces
lock-step releases.

Duplicating a helper between two packages is the intended answer. This hook exists
so that choice stays deliberate.

Benchmark baselines are checked too — they ship in the sdist and are part of what a
contributor reads. Tests and examples are checked as well, since a cross-import
there still means the monorepo layout is doing work that a user's install cannot.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES_DIR = REPO_ROOT / "packages"


def package_names() -> set[str]:
    """Return the import name of every package in the repo."""
    return {
        p.name for p in PACKAGES_DIR.iterdir() if p.is_dir() and (p / "pyproject.toml").exists()
    }


def imported_roots(tree: ast.Module) -> set[str]:
    """Return the top-level module name of every import in ``tree``."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def main() -> int:
    """Scan every package for imports of a sibling package."""
    names = package_names()
    if not names:
        return 0

    violations: list[str] = []
    for pkg in sorted(names):
        siblings = names - {pkg}
        for py in sorted((PACKAGES_DIR / pkg).rglob("*.py")):
            if any(part in {".venv", "target", "__pycache__"} for part in py.parts):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                violations.append(f"{py.relative_to(REPO_ROOT)}: could not parse ({exc})")
                continue
            violations.extend(
                f"{py.relative_to(REPO_ROOT)}: imports sibling package '{offending}'"
                for offending in sorted(imported_roots(tree) & siblings)
            )

    for line in violations:
        print(line, file=sys.stderr)
    if violations:
        print(
            "\nPackages publish independently and must not import each other. "
            "Duplicate the helper instead.",
            file=sys.stderr,
        )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
