#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="MatDBForge",
    version="0.1",
    description="MatDBForge Test",
    author="Pol Sanz",
    author_email="me@polsanz.xyz",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
