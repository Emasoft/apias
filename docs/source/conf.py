import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'APIAS'
copyright = '2024, Emanuele Sabetta'
author = 'Emanuele Sabetta'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
add_module_names = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
