.. -*- mode: rst -*-

|Travis|_ |PyPi|_

.. |Travis| image:: https://travis-ci.com/ocampor/image-quality.svg?branch=master
.. _Travis: https://travis-ci.com/ocampor/image-quality

.. |PyPi| image:: https://img.shields.io/pypi/dm/image-quality?color=blue   :alt: PyPI - Downloads
.. _PyPi: https://pypi.org/project/image-quality/

Image Quality
=============

Description
-----------

Image quality is an open source software library for Automatic Image
Quality Assessment (IQA).

Dependencies
------------

-  Python 3.8
-  (Development) Docker

Installation
------------

The package is public and is hosted in PyPi repository. To install it in
your machine run

::

   pip install image-quality

Example
-------

After installing ``image-quality`` package, you can test that it was
successfully installed running the following commands in a python
terminal.

::

   >>> import imquality.brisque as brisque
   >>> import PIL.Image

   >>> path = 'path/to/image'
   >>> img = PIL.Image.open(path)
   >>> brisque.score(img)
   4.9541572815704455


Development
-----------

In case of adding a new tensorflow dataset or modifying the location of a zip file, it is
necessary to update the url checksums. You can find the instructions in the following
`tensorflow documentation <https://www.tensorflow.org/datasets/add_dataset#1_adjust_the_checksums_directory>`_.

The steps to create the url checksums are the following:

1. Take the file with the dataset configuration (e.g. live_iqa.py) an place it in the ``tensorflow_datasets``
folder. The folder is commonly placed in ``${HOME}/.local/lib/python3.8/site-packages`` if you
install the python packages using the ``user`` flag.

2. Modify the ``__init__.py`` of the ``tensorflow_datasets`` to import your new dataset.
For example ``from .image.live_iqa import LiveIQA`` at the top of the file.

3. In your terminal run the commands:
::

   touch url_checksums/live_iqa.txt
   python -m tensorflow_datasets.scripts.download_and_prepare  \
      --register_checksums  \
      --datasets=live_iqa

4. The file ``live_iqa.txt`` is going to contain the checksum. Now you can copy and paste it to your
project's ``url_checksums`` folder.

Sponsor
-------

.. image:: https://github.com/antonreshetov/mysigmail/raw/master/jetbrains.svg?sanitize=true
   :target: <https://www.jetbrains.com/?from=mysigmail>_

Maintainer
----------

- `Ricardo Ocampo <https://ocampor.com>`_
