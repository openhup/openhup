"""Backend test package.

Made a package so test modules can share `conftest` helpers via relative imports
(`from .conftest import T0, samples`) rather than duplicating clock fixtures.
"""
