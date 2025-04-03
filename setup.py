from setuptools import setup, find_packages

setup(
    name="feynman",
    version="0.1.0",
    description="A universal DSL for classical and quantum physics simulations",
    author="Feynman Team",
    packages=find_packages(),
    install_requires=[
        "lark>=1.1.5",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "sympy>=1.10.0",
        "matplotlib>=3.5.0",
    ],
    entry_points={
        "console_scripts": [
            "feynman=feynman.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
) 