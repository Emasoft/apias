Installation
===========

Requirements
-----------

APIAS requires **Python 3.10 or later**. It also depends on several other Python packages,
which are automatically installed when you install APIAS.

.. note::
   Python 3.9 support was dropped in version 0.1.16 due to the use of
   ``dataclass(kw_only=True)`` which was introduced in Python 3.10.

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
