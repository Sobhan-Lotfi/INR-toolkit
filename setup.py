from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inr-toolkit",
    version="0.1.0",
    author="INR Toolkit Contributors",
    description="Educational toolkit for Implicit Neural Representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sobhan-Lotfi/INR-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "pytest>=6.0.0",
            "black>=21.0",
        ],
    },
)
