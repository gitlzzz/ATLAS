"""Sphinx configuration file."""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib as pl
import re
import subprocess
import sys

import MatDBForge.core.code_utils as mdb_cud


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


autodoc_mock_imports = ["aiida"]


def get_whitelist_pattern(version_list: list) -> str:
    """
    Generates a regex pattern for sphinx-multiversion based on a list of versions.

    Parameters
    ----------
    version_list : list
        A list of version objects or strings.
        ```
        versions = [<Version('0.45.0')>, <Version('0.44.2')>, ...]
        ```

    Returns
    -------
    str
        A regex string combining all versions with OR operators.
    """
    cleaned_versions = []
    for v in version_list:
        # Convert objects to strings
        v_str = str(v)

        # IMPORTANT: If my Git tags are 'v0.45.0' but the Version object is '0.45.0',
        # I must prepend the 'v' here.
        # Reenable if I end up switching to 'v' tags.
        # if not v_str.startswith("v"):
        #     v_str = f"v{v_str}"

        # Escape special regex characters (like '.')
        cleaned_versions.append(re.escape(v_str))

    # Join them with the OR operator '|'
    combined_pattern = "|".join(cleaned_versions)

    # Wrap in start (^) and end ($) anchors to ensure exact matches
    return f"^({combined_pattern})$"


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
release = "0.46.6"


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

# This defines a new role named "alt"
# When used, it will add the CSS class "code-alt" to the element
myst_prolog = """
.. role:: alt
   :class: code-alt

.. role:: codeheader
   :class: code-header
"""

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
    "active_learning.py",
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
    "github_url": "https://github.com/pol-sb/MatDBForge",
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

# --- sphinx-multiversion configuration ---
#
# This will build the 'master' branch AND all tags that
# start with 'v', while ignoring all other branches.

newest_tag, git_hash = mdb_cud.get_last_tagged_version_local(
    "/home/runner/work/MatDBForge/MatDBForge"
)
tags = mdb_cud.get_list_of_tags("/home/runner/work/MatDBForge/MatDBForge")

last_10_versions = tags[:10]

smv_tag_whitelist = rf"{get_whitelist_pattern(last_10_versions)}"
smv_branch_whitelist = r"^master$"
