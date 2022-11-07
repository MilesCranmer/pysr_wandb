#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "pysr==0.11.9",
    "wandb>=0.13.0",
    "numpy",
    "pandas",
]

test_requirements = []

setup(
    author="Miles Cranmer",
    author_email="miles.cranmer@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Experiments in tuning PySR with W&B",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pysr_wandb",
    name="pysr_wandb",
    packages=find_packages(include=["pysr_wandb", "pysr_wandb.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/MilesCranmer/pysr_wandb",
    version="0.1.0",
    zip_safe=False,
)
