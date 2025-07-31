"""
Setup script for GraphMind

Distributed Graph Neural Networks with Byzantine Fault Tolerant Consensus
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [req for req in requirements if req and not req.startswith('#')]

setup(
    name="graphmind",
    version="0.1.0",
    author="Ayomide Caleb Adekoya",
    author_email="",
    description="Distributed Graph Neural Networks with Byzantine Fault Tolerant Consensus",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/graphmind",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/graphmind/issues",
        "Documentation": "https://github.com/yourusername/graphmind/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "pytest-benchmark>=4.0.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.1",
            "flake8>=6.1.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "visualization": [
            "matplotlib>=3.8.2",
            "seaborn>=0.13.0",
            "plotly>=5.17.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.3.0",
            "pytorch-lightning>=2.1.2",
            "accelerate>=0.24.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "graphmind-train=distributed_train:main",
            "graphmind-byzantine=byzantine_simulation:main",
            "graphmind-test=tests.run_tests:run_all_tests",
        ],
    },
    include_package_data=True,
    package_data={
        "graphmind": [
            "config/*.yaml",
            "data/.gitkeep",
            "benchmarks/*.py",
        ],
    },
)