#!/usr/bin/env python

"""The setup script."""

import io
from os import path as op
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()


with open("HISTORY.rst") as history_file:
    history = history_file.read()

here = op.abspath(op.dirname(__file__))

# get the dependencies and installs

requirements = ["cftime>=1.6.0",
    "eofs>=1.4",
    "geopandas>=0.12",
    "numpy>=1.21",
    "pandas>=1.3",
    "scikit-learn>=0.24",
    "scikit-optimize>=0.9.0",
    "scipy>=1.7",
    "seaborn>=0.11",
    "tensorflow>=2.8.0",
    "xarray>=2023.1",
    "xgboost>=1.5",
    "cycler>=0.10"
    ]
dev_requirements = []
with open("requirements_dev.txt") as dev:
    for dependency in dev.readlines():
        dev_requirements.append(dependency)

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]
KEYWORDS = "pyESD Empirical Statistical Downscaling"

setup(
    author="Daniel Boateng",
    author_email='dannboateng@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Python Package for Empirical Statistical Downscaling. pyESD is under active development and all colaborators are welcomed. The purpose of the package is to downscale any climate variables e.g. precipitation and temperature using predictors from  reanalysis datasets (eg. ERA5) to point scale. pyESD adopts many ML and AL as the transfer function. ",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords=KEYWORDS,
    name='pyESD',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Dan-Boat/PyESD',
    extras_require={"dev": dev_requirements},
    version='1.0.7',
    zip_safe=False,
)
