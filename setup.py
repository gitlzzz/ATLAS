#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="MatDBForge",
    version="0.3.0",
    description="MatDBForge",
    author="Pol Sanz",
    author_email="me@polsanz.xyz",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        "aiida.calculations": [
            "mace-train = MatDBForge.active_learning.mace_tools_aiida:TrainMACEModelCalculation",
            "mace-get-descriptors = MatDBForge.active_learning.mace_tools_aiida:GetMACEDescriptorsCalculation",
        ],
        "aiida.calculations.monitors": [
            "monitor.davwarning = MatDBForge.workflows.monitors:output_monitor"
        ],
        "aiida.workflows": [
            "mdb-active-learning = MatDBForge.workflows.active_learning:ActiveLearningWorkChain",
            "mdb-active-learning-base = MatDBForge.workflows.active_learning:ActiveLearningBaseWorkChain",
        ],
        "aiida.parsers": [
            "mace-training-parser = MatDBForge.active_learning.mace_tools_aiida:TrainMACEModelCalculationParser",
            "mace-descriptors-parser = MatDBForge.active_learning.mace_tools_aiida:GetMACEDescriptorsCalculationParser",
        ],
        "console_scripts": [
            "mdb_active_learning=MatDBForge.core.command_line:run_active_learning",
            "mdb_conf_gen=MatDBForge.core.command_line:gen_default_config",
            "mdb_monitor_al_loop=MatDBForge.core.command_line:monitor_al_loop",
        ],
    },
)
