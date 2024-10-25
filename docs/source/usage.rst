Usage
=====

Basic Usage
----------

APIAS can be used both as a command-line tool and as a Python library.

Command Line Interface
--------------------

To scrape documentation from a URL:

.. code-block:: bash

   apias scrape https://api.example.com/docs

To convert documentation to a specific format:

.. code-block:: bash

   apias convert input.html --format markdown --output api_docs.md

Python API
---------

Here's how to use APIAS in your Python code:

.. code-block:: python

   from apias import apias

   # Scrape documentation from a URL
   doc = apias.scrape_url("https://api.example.com/docs")

   # Convert to markdown
   markdown = doc.to_markdown()

   # Save to file
   doc.save("api_docs.md")

Configuration
------------

APIAS can be configured using a configuration file or through the API:

.. code-block:: python

   config = {
       "format": "markdown",
       "output": "api_docs.md",
       "base_url": "https://api.example.com"
   }

   apias.scrape_and_save("https://api.example.com/docs", config)
