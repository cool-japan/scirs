# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "SciRS2 Python"
copyright = "2026, COOLJAPAN OU (Team Kitasan)"
author = "Team Kitasan"

# The short X.Y version
version = "0.4"

# The full version, including alpha/beta/rc tags
release = "0.4.3"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",       # Auto-doc from docstrings
    "sphinx.ext.napoleon",      # Google/NumPy docstring styles
    "sphinx.ext.viewcode",      # Source links in docs
    "sphinx.ext.intersphinx",   # Cross-project links
    "sphinx.ext.autosummary",   # Auto-generate summary tables
    "sphinx.ext.doctest",       # Doctest runner
]

# Napoleon settings — we use NumPy docstring format
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# ---------------------------------------------------------------------------
# Intersphinx mapping
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/cool-japan/scirs",
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "footer_items": ["copyright", "sphinx-version"],
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "navigation_depth": 3,
}

html_title = f"SciRS2 Python {release}"

html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# Extension settings
# ---------------------------------------------------------------------------

# Display TODO items
todo_include_todos = False

# Autosummary: generate stub pages automatically
autosummary_generate = True
