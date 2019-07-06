from distutils.core import setup

setup(
    name='image-quality',
    packages=['image-quality'],
    version='0.1.0-a0',
    license='Apache 2.0',
    description="""
    ## Project Description
    
    Image Quality is an open source software library for Automatic Image Quality Assessment (IQA). 
    """,
    author='Ricardo Ocampo',
    author_email='me@ocampor.ai',
    url='https://github.com/ocampor/image-quality',  # Provide either the link to your github or to your website
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',  # I explain this later on
    keywords=['image', 'quality', 'reference', 'reference-less'],  # Keywords that define your package best
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
        'License :: OSI Approved :: Apache 2.0',
        'Programming Language :: Python :: 3.7',
    ],
    include_package_data=True
)
