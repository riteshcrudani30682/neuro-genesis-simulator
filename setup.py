#!/usr/bin/env python3
"""
Setup script for Neuro-Genesis Simulator
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuro-genesis-simulator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A neural network cellular automaton simulator with reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuro-genesis-simulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "neuro-genesis=main:run_with_control_panel",
        ],
    },
    keywords="neural-network cellular-automaton reinforcement-learning simulation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/neuro-genesis-simulator/issues",
        "Source": "https://github.com/yourusername/neuro-genesis-simulator",
    },
)
