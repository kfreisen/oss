"""Type stubs for the compiled kernel.

Hand-maintained, because the extension is a shared object: mypy cannot infer these,
and griffe (which the docs site uses) cannot import them. This file is the single
declaration of the FFI surface — keep it in step with `rust/src/lib.rs`.

Nothing here is public API. Use the wrappers in :mod:`slatekit`.
"""

__version__: str

def active_isa() -> str:
    """Return the SIMD path selected at runtime: ``"avx2"`` or ``"scalar"``."""
