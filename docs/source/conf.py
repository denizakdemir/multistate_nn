# Configuration file for the Sphinx documentation builder.
import os
import sys
import sphinx_rtd_theme

# Add the project root directory to the path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'MultiStateNN'
copyright = '2024, Deniz Akdemir'
author = 'Deniz Akdemir'
version = '0.4.0'
release = '0.4.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'MultiStateNNdoc'

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}