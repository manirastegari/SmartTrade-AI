#!/usr/bin/env python3
"""
Setup script for SmartTrade AI
Advanced Trading Analysis System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements_minimal.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="smarttrade-ai",
    version="2.0.0",
    author="Mani Rastegari",
    author_email="mani.rastegari@gmail.com",
    description="Advanced AI-powered trading analysis system with 500+ stocks and 50+ technical indicators",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smarttrade-ai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/smarttrade-ai/issues",
        "Source": "https://github.com/yourusername/smarttrade-ai",
        "Documentation": "https://github.com/yourusername/smarttrade-ai#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Environment :: Web Environment",
        "Framework :: Streamlit",
    ],
    python_requires=">=3.12",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "smarttrade=simple_trading_analyzer:main",
            "smarttrade-enhanced=enhanced_trading_app:main",
        ],
    },
    keywords=[
        "trading",
        "stock analysis",
        "technical analysis",
        "fundamental analysis",
        "machine learning",
        "artificial intelligence",
        "finance",
        "investment",
        "yfinance",
        "streamlit",
        "free",
        "open source",
    ],
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    platforms=["macOS", "Linux"],
    license="MIT",
    maintainer="Mani Rastegari",
    maintainer_email="mani.rastegari@gmail.com",
)
