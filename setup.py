#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="MatDBForge",
    version="0.2.4",
    description="MatDBForge",
    author="Pol Sanz",
    author_email="me@polsanz.xyz",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        "aiida.calculations": [
            "mace-train = MatDBForge.workflows.active_learning:TrainMACEModelCalculation"
        ],
        "aiida.calculations.monitors": [
            "monitor.davwarning = MatDBForge.workflows.monitors:output_monitor"
        ],
        "aiida.workflows": [
            "mdb-active-learning = MatDBForge.workflows.active_learning:ActiveLearningWorkChain",
            "mdb-active-learning-base = MatDBForge.workflows.active_learning:ActiveLearningBaseWorkChain",
        ],
        "aiida.parsers": [
            "mace-training-parser = MatDBForge.workflows.active_learning:TrainMACEModelCalculationParser",
        ],
    },
)
