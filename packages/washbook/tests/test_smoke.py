"""Scaffold-level checks: the package imports and reports a version.

These stay after washbook is implemented. An import failure here is the fastest possible
signal that a build, a lock, or a wheel is broken, and it runs in milliseconds.
"""

import washbook


def test_version_is_exposed():
    assert isinstance(washbook.__version__, str)
    assert washbook.__version__
