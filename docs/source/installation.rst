Installation
===========

Requirements
-----------

APIAS requires Python 3.9 or later. It also depends on several other Python packages,
which are automatically installed when you install APIAS.

Installing APIAS
---------------

You can install APIAS using pip:

.. code-block:: bash

   pip install apias

Development Installation
----------------------

If you want to contribute to APIAS or install it with development dependencies:

.. code-block:: bash

   git clone https://github.com/Emasoft/apias.git
   cd apias
   pip install -e ".[dev,test,docs]"
