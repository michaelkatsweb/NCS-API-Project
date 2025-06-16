#!/usr/bin/env python3
"""
NeuroCluster Streamer Python SDK - Package Setup
==============================================
Official Python client library for the NCS API

Author: NCS API Development Team
Year: 2025
"""

import os
import re

from setuptools import find_packages, setup


# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "ncs_client", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Read long description from README
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Official Python client library for the NeuroCluster Streamer API"


# Read requirements from requirements.txt
def get_requirements(filename="requirements.txt"):
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    # Package metadata
    name="ncs-python-sdk",
    version=get_version(),
    description="Official Python client library for the NeuroCluster Streamer API",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    # Author information
    author="NCS API Development Team",
    author_email="sdk@yourdomain.com",
    maintainer="NCS API Development Team",
    maintainer_email="sdk@yourdomain.com",
    # Project URLs
    url="https://github.com/your-org/ncs-api",
    project_urls={
        "Documentation": "https://docs.ncs-api.com/sdk/python",
        "Source Code": "https://github.com/your-org/ncs-api/tree/main/sdk/python",
        "Issue Tracker": "https://github.com/your-org/ncs-api/issues",
        "API Documentation": "https://api.yourdomain.com/docs",
        "Changelog": "https://github.com/your-org/ncs-api/blob/main/CHANGELOG.md",
    },
    # Package configuration
    packages=find_packages(exclude=["tests*", "examples*"]),
    package_dir={"": "."},
    include_package_data=True,
    # Dependencies
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": get_requirements("requirements-dev.txt"),
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "httpx[test]>=0.25.0",
            "responses>=0.23.0",
            "factory-boy>=3.2.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "myst-parser>=0.18.0",
        ],
        "performance": [
            "numpy>=1.24.0",
            "pandas>=1.5.0",
            "pyarrow>=10.0.0",
        ],
        "async": [
            "aiofiles>=22.0.0",
            "asyncio-throttle>=1.0.0",
        ],
    },
    # Python version requirements
    python_requires=">=3.8",
    # Package classification
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        # License
        "License :: OSI Approved :: MIT License",
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        # Operating System
        "Operating System :: OS Independent",
        # Framework
        "Framework :: AsyncIO",
        # Typing
        "Typing :: Typed",
    ],
    # Keywords for discovery
    keywords=[
        "api-client",
        "clustering",
        "machine-learning",
        "data-science",
        "stream-processing",
        "neurocluster",
        "real-time",
        "analytics",
        "artificial-intelligence",
        "http-client",
    ],
    # Entry points for CLI tools (if needed)
    entry_points={
        "console_scripts": [
            "ncs-cli=ncs_client.cli:main",
        ],
    },
    # Package data
    package_data={
        "ncs_client": [
            "py.typed",  # PEP 561 typed package marker
            "config/*.json",
            "templates/*.json",
        ],
    },
    # Data files
    data_files=[
        (
            "share/ncs-python-sdk/examples",
            [
                "examples/basic_usage.py",
                "examples/streaming_example.py",
                "examples/batch_processing.py",
            ],
        ),
    ],
    # Options
    zip_safe=False,  # Required for mypy to find py.typed
    # Additional metadata for modern Python packaging
    license="MIT",
    platforms=["any"],
    # Test configuration
    test_suite="tests",
    tests_require=get_requirements("requirements-test.txt"),
    # Build options
    options={
        "build_scripts": {
            "executable": "/usr/bin/env python3",
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
)

# Post-install message
print(
    """
🎉 NeuroCluster Streamer Python SDK installed successfully!

Quick start:
    from ncs_client import NCSClient
    
    client = NCSClient(
        base_url="https://api.yourdomain.com",
        api_key="your-api-key"
    )
    
    result = client.process_points([[1, 2, 3]])

📚 Documentation: https://docs.ncs-api.com/sdk/python
🐛 Issues: https://github.com/your-org/ncs-api/issues
💬 Support: sdk@yourdomain.com

Happy clustering! 🚀
"""
)
