"""
Sphinx Configuration for CapibaraGPT Documentation
===================================================

This file configures Sphinx to generate API documentation automatically
from Python docstrings.

Usage:
    cd docs
    make html
    # or
    sphinx-build -b html . _build/html
"""

import os
import sys
from datetime import datetime

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'CapibaraGPT'
copyright = f'{datetime.now().year}, CapibaraGPT Team'
author = 'CapibaraGPT Team'
version = '3.0'
release = '3.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx.ext.napoleon',          # Support Google/NumPy docstrings
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to other projects' docs
    'sphinx.ext.todo',              # Support TODOs
    'sphinx.ext.coverage',          # Check documentation coverage
    'sphinx.ext.mathjax',           # Math support
    'sphinx.ext.inheritance_diagram',  # Class inheritance diagrams
    'myst_parser',                  # Markdown support
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**/__pycache__',
    '**/test_*.py',
]

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980b9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

# Mock imports for modules that might not be available
autodoc_mock_imports = [
    'torch',
    'jax',
    'flax',
    'tensorflow',
    'numpy',
    'scipy',
    'pandas',
    'psutil',
    'transformers',
    'datasets',
    'accelerate',
    'wandb',
    'tensorboard',
    'optuna',
    'ray',
    'dask',
]

# -- Options for autosummary -------------------------------------------------

autosummary_generate = True
autosummary_imported_members = True

# -- Options for Napoleon (Google/NumPy docstrings) --------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True

# -- Options for MyST parser (Markdown) --------------------------------------

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]

myst_heading_anchors = 3

# -- Custom setup ------------------------------------------------------------

def setup(app):
    """Custom Sphinx setup."""
    # Create _static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(__file__), '_static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Create custom.css if it doesn't exist
    css_file = os.path.join(static_dir, 'custom.css')
    if not os.path.exists(css_file):
        with open(css_file, 'w') as f:
            f.write("""/* Custom styles for CapibaraGPT documentation */

/* Improve code block styling */
.highlight {
    background: #f8f8f8;
    border-radius: 4px;
}

/* Better table styling */
table.docutils {
    border-collapse: collapse;
    width: 100%;
}

table.docutils td, table.docutils th {
    padding: 8px 12px;
    border: 1px solid #e1e4e5;
}

/* Method signatures */
.sig-name {
    font-weight: 600;
}

/* Admonitions */
.admonition {
    border-radius: 4px;
}

/* API reference styling */
dl.py.class > dt,
dl.py.function > dt,
dl.py.method > dt {
    background: #f0f0f0;
    padding: 6px 10px;
    border-radius: 4px;
    margin-bottom: 10px;
}
""")
