from distutils.core import setup

setup(
    name='image-quality',
    packages=['image-quality'],
    version='0.1.1-a0',
    license='Apache 2.0',
    description='Image quality is an open source software library for Automatic Image Quality Assessment (IQA).',
    author='Ricardo Ocampo',
    author_email='me@ocampor.ai',
    url='https://github.com/ocampor/image-quality',
    download_url='https://github.com/ocampor/image-quality/archive/0.1.0-a0.tar.gz',
    keywords=['image', 'quality', 'reference', 'reference-less'],
    install_requires=[
        'Pillow==5.2.0',
        'numpy==1.15.1',
        'scipy==1.1.0',
        'opencv-python==3.4.2.17',
        'scikit-image==0.14.0',
    ],
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
