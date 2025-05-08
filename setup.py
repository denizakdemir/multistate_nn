from setuptools import setup, find_packages

setup(
    name="multistate_nn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.23.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "networkx>=3.0",
    ],
    extras_require={
        "bayesian": ["pyro-ppl>=1.9.0"],
        "dev": ["pytest>=7.3.1", "jupyter>=1.0.0", "scikit-learn>=1.0.0"],
        "examples": ["pymsm>=0.1.0", "scikit-learn>=1.0.0"],
    },
    python_requires=">=3.9",
    author="Deniz Akdemir",
    author_email="denizakdemir@gmail.com",
    description="Discrete-time multistate models as neural network heads",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/denizakdemir/multistate_nn",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
