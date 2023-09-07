#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="MatDBForge",
    version="0.2a",
    description="MatDBForge",
    author="Pol Sanz",
    author_email="me@polsanz.xyz",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        "aiida.calculations.monitors": [
            "monitor.davwarning = MatDBForge.workflows.monitors:output_monitor"
        ]
    },
)
