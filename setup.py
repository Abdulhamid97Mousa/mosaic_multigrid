"""Shim for editable installs and legacy tooling.

All package metadata is defined in pyproject.toml.  This file exists
only so that ``pip install -e .`` works with older pip versions that
do not fully support PEP 660.
"""
from setuptools import setup

setup()
