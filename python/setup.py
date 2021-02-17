import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "gpu-rsfk",
    version = "0.0.0",
    author = "Bruno Meyer",
    author_email = "buba.meyer_@hotmail.com",
    description = "GPU implementation of Random Projection Forest KNN (Similarity Search)",
    # license = "BSD",
    # keywords = "example documentation tutorial",
    # url = "http://packages.python.org/an_example_pypi_project",
    packages=['gpu_rsfk'],
    package_data={'gpu_rsfk': ['librsfk.so']},
    install_requires=[
        'numpy >= 1.14.1',
    ],
    # long_description=read('README'),
    classifiers=[
        "Development Status :: 1 - Planning",
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=[
        'Similarity Search',
        'KNN',
        'GPU',
        'CUDA',
        'Machine Learning',
        'AI'
    ],
    zip_safe = False # Prevent egg-link creation
)