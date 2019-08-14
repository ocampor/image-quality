# Image Quality

## Description

Image quality is an open source software library for Automatic Image Quality Assessment (IQA).

## Dependencies

- Python 3.7
- LibSVM
- (Optional) Docker

## Installation

The package is public and is hosted in PyPi repository. To install it in your machine run
```.env
pip install imquality
```

## Example

After installing `imquality` package, you can test that it was successfully installed running the
following commands in a python terminal.
```.python
>>> import imquality.brisque as brisque
>>> import PIL.Image

>>> path = 'path/to/image'
>>> img = PIL.Image.open(path)
>>> brisque.score(img)
4.9541572815704455
```

## Report Bugs

## Maintainer
- Ricardo Ocampo - [me@ocampor.ai](me@ocampor.ai)
 