#!/usr/bin/env python

from distutils.core import setup

setup(
    name="pydynamics",
    version="0.0.1",
    description="Package for fitting dynamical systems models to timeseries data",
    author="Ali Zaidi",
    author_email="zaidi@icord.org",
    package_dir={"pydynamics": "src/pydynamics"},
    packages=["pydynamics"],
)
