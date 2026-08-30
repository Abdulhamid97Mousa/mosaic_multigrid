# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add source code to path for autodoc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print("[DOCS] mosaic_multigrid library path: {}".format(sys.path[0]))

# -- Project information -----------------------------------------------------

project = "mosaic_multigrid"
copyright = "2026, mosaic_multigrid Contributors"
author = "Abdulhamid M. Mousa"
release = "7.0.0"

# The master toctree document.
master_doc = "index"

# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # External extensions
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinx_favicon",
    "sphinxcontrib.mermaid",
]

# Napoleon settings (for Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# MyST parser extensions
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "deflist",
    "linkify",
]

# Mock imports for autodoc (heavy or system-specific packages)
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "gymnasium",
    "gym",
    "pygame",
    "numba",
    "pettingzoo",
    "aenum",
]

# Pygments style
pygments_style = "tango"

# Templates and exclusions
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "mosaic_multigrid"
html_short_title = "mosaic_multigrid"
html_scaled_image_link = False
html_static_path = ["_static"]

html_theme_options = {
    # Logo (mosaic brand)
    "light_logo": "figures/logo.png",
    "dark_logo": "figures/logo.png",
    # Source links
    "source_repository": "https://github.com/Abdulhamid97Mousa/mosaic_multigrid",
    "source_branch": "main",
    "source_directory": "docs/source",
    "top_of_page_buttons": ["view", "edit"],
    # Navigation
    "navigation_with_keys": True,
}

html_css_files = [
    "css/custom.css",
]

# Favicons (multiple sizes for different browsers / bookmark widgets)
favicons = [
    {"rel": "icon", "type": "image/x-icon", "href": "figures/favicon.ico"},
    {"rel": "icon", "type": "image/png", "sizes": "16x16", "href": "figures/favicon_16.png"},
    {"rel": "icon", "type": "image/png", "sizes": "32x32", "href": "figures/favicon_32.png"},
    {"rel": "icon", "type": "image/png", "sizes": "48x48", "href": "figures/favicon_48.png"},
    {"rel": "icon", "type": "image/png", "sizes": "64x64", "href": "figures/favicon_64.png"},
]

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "pettingzoo": ("https://pettingzoo.farama.org/", None),
}
