"""Scaffold-level checks: the package imports and reports a version.

These stay after impliedmove is implemented. An import failure here is the fastest possible
signal that a build, a lock, or a wheel is broken, and it runs in milliseconds.
"""

import impliedmove


def test_version_is_exposed():
    assert isinstance(impliedmove.__version__, str)
    assert impliedmove.__version__
