"""
Setup script for TriGraphX package.

For pure Python installation:
    pip install .

For Rust acceleration (requires maturin):
    pip install maturin
    maturin develop --release

Or use pyproject.toml with maturin backend:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="trigraphx",
    version="0.1.0",
    description="TriGraphX - Unified Metric Space Database combining tree, graph, and vector storage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TriGraphX Contributors",
    url="https://github.com/novaos/trigraphx",
    packages=find_packages(exclude=["tests*", "trigraphx_core*", "trigraphx_rust*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "cryptography>=38.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "mypy>=0.950",
        ],
        "rust": [
            "maturin>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
