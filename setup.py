from distutils.core import setup

setup(
    name='image-quality',
    packages=['imquality'],
    version='1.0.1',
    license='Apache 2.0',
    description='Image quality is an open source software library for Automatic Image Quality Assessment (IQA).',
    author='Ricardo Ocampo',
    author_email='me@ocampor.ai',
    url='https://github.com/ocampor/image-quality',
    keywords=['image', 'quality', 'reference', 'reference-less'],
    install_requires=[
        'Pillow==5.2.0',
        'numpy==1.16.4',
        'scipy==1.3.0',
        'opencv-python==4.1.0.25',
        'scikit-image==0.15.0',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True
)
