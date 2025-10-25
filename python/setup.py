import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    readme_path = os.path.join(os.path.dirname(__file__), '..', fname)
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name = "gpu-rsfk",
    version = "0.0.1",
    author = "Bruno Meyer",
    author_email = "buba.meyer_@hotmail.com",
    description = "GPU implementation of Random Sample Forest KNN (Similarity Search)",
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    license = "MIT",
    url = "https://github.com/BrunoMeyer/gpu-rsfk",
    project_urls={
        "Bug Tracker": "https://github.com/BrunoMeyer/gpu-rsfk/issues",
        "Source Code": "https://github.com/BrunoMeyer/gpu-rsfk",
    },
    packages=['gpu_rsfk'],
    package_data={'gpu_rsfk': ['librsfk.so']},
    install_requires=[
        'numpy >= 1.14.1',
    ],
    python_requires='>=3.6',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords=[
        'Similarity Search',
        'KNN',
        'GPU',
        'CUDA',
        'Machine Learning',
        'AI',
        'Random Sample Forest',
        'Nearest Neighbors'
    ],
    zip_safe = False # Prevent egg-link creation
)