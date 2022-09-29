# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:50:59 2022

@author: dboateng
"""

from setuptools import setup
import sys
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
      name = "pyESD",
      version = "0.0.1",
      description= "Python package for Empirical Statistical Downscaling",
      long_description = README,
      long_description_content_type = "text/markdown",
      keywords = "climate downscaling", "statistical downscaling", "scikit-learn", "xarray", 
      url="https://github.com/Dan-Boat/PyESD",
      author="Daniel Boateng",
      author_email= "dannboateng@gmail.com",
      license="MIT",
      packages=["pyESD"],
      install_requires=["mpi4py",
                        "numpy", 
                        "pandas",
                        "xarray", 
                        "statsmodels",
                        "seaborn",
                        "sklearn",
                        "scipy",
                        "eofs",
                        ],

      classifiers = [
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5', 
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8', 
        'Programming Language :: Python :: 3.9',
        
          ],
      
      include_package_data = True,
      entry_points = {}
      
      )
