Welcome to APIAS documentation!
==============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   contributing
   changelog

Overview
--------

APIAS (AI Powered API Documentation Scraper) is a powerful tool that helps you extract
and convert API documentation from various sources into structured formats.

Quick Start
----------

Installation:

.. code-block:: bash

   pip install apias

Basic usage:

.. code-block:: python

   from apias import apias

   # Basic usage
   doc = apias.scrape_url("https://api.example.com/docs")
   print(doc.to_markdown())

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
