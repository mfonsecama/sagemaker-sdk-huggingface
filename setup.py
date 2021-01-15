# Lint as: python3
from setuptools import find_packages
from setuptools import setup

# Declare minimal set for installation
required_packages = [
    "sagemaker",
    "sagemaker[local]",
    "sagemaker-experiments",
    "torch==1.6.0",
    "transformers",
    "datasets",
    "sklearn",
    "boto3",
    "numpy>=1.17.0",
    "matplotlib",
]

# Specific use case dependencies
extras = {}

# dependency for code quality
extras["quality"] = [
    "black",
    "isort",
    "flake8==3.7.9",
]

# Meta dependency groups
extras["all"] = [item for group in extras.values() for item in group]

# Tests specific dependencies (do not need to be included in 'all')
extras["test"] = [
    extras["all"],
    "pytest",
    "pytest-xdist",
]


setup(
    name="sagemakerhuggingface",
    version="0.0.1",
    description="Custom Sdk implementation for the huggingface libraries",
    long_description=open("readme.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Philipp",
    author_email="schmidphilipp1995@gmail.com",
    url="https://github.com/philschmid/sagemaker-sdk-huggingface/tree/SagemakerTrainer",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=required_packages,
    extras_require=extras,
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="aws machine learning sagemaker metrics",
)