# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dl-utils'
copyright = '2025, Yaojie Shen'
author = 'Yaojie Shen'
release = 'https://github.com/AcherStyx/Deep-Learning-Utils'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.programoutput"
]


# Make repo root importable for autodoc (useful for local builds).
import os
import sys

_DOCS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_DOCS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



autodoc_member_order = 'bysource'

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
