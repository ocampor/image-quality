import os

from setuptools import setup

import imquality

DESCRIPTION = 'Image quality is an open source software library for Automatic Image Quality Assessment (IQA).'
LICENSE = 'Apache 2.0'
DIST_NAME = 'image-quality'
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
AUTHOR_EMAIL = 'me@ocampor.ai'
AUTHOR = 'Ricardo Ocampo'
URL = 'https://github.com/ocampor/image-quality'
VERSION = imquality.__version__
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/ocampor/image-quality/issues',
    'Source Code': 'https://github.com/ocampor/image-quality'
}
LONG_DESCRIPTION_TYPE = 'text/x-rst'

with open(os.path.join(ROOT_PATH, 'README.rst'), 'r') as readme:
    LONG_DESCRIPTION = readme.read()

setup(
    name=DIST_NAME,
    packages=['imquality'],
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    keywords=['image', 'quality', 'reference', 'reference-less'],
    install_requires=[
        'Pillow>=5.2.0',
        'numpy>=1.16.4',
        'scipy>=1.3.0',
        'scikit-image>=0.15.0',
        'libsvm>=3.23.0',
    ],
    extras_require={
        'dev': [
            'pytest>=4.4.0',
            'pytest-xdist',
        ],
        'dataset': [
            'tensorflow>=2.0.0',
            'tensorflow-datasets>=1.2.0',
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux'
    ],
    include_package_data=True,
    python_requires=">=3.6",
    project_urls=PROJECT_URLS,
)
