"""Sphinx configuration file."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import pathlib as pl
import subprocess
import sys


def get_git_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except subprocess.CalledProcessError:
        commit_hash = "unknown"
    return commit_hash


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, pl.Path("../src/MatDBForge").absolute())


# -- Project information -----------------------------------------------------

project = "MatDBForge"
copyright = "2024, Pol Sanz"
author = "Pol Sanz"

# The full version, including alpha/beta/rc tags
release = "0.20.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx_multiversion",
    #    'sphinx.ext.viewcode',
    #    'sphinx_autodoc_typehints'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "benchmarks/*",
    "tests/*",
    "devel/*",
    "display_db/*",
    "*sync-conflict*",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

myst_enable_extensions = ["colon_fence", "html_image", "dollarmath"]


html_favicon = "./source/favicon.svg"
html_logo = "../media/logo_dark.png"

html_theme_options = {
    "logo_only": True,
    "version_selector": True,
    "display_version": True,
    "github_url": "https://github.com/pol-sb/MatDBForge"
}

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "pol-sb",  # Username
    "github_repo": "MatDBForge",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
    "display_lower_left": True,
}

html_context["commit_hash"] = get_git_commit_hash()
