import os
from setuptools import setup

# This setup.py provides explicit configuration for backward compatibility
# with legacy installation methods (setup.py install).
# Modern build tools will use pyproject.toml instead.

setup(
    name="gpu-rsfk",
    version="0.0.1",
    author="Bruno Meyer",
    author_email="buba.meyer_@hotmail.com",
    description="GPU implementation of Random Sample Forest KNN (Similarity Search)",
    license="MIT",
    url="https://github.com/BrunoMeyer/gpu-rsfk",
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
    zip_safe=False  # Prevent egg-link creation
)