from setuptools import setup, find_packages

setup(
    name="tensor_dump",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "tensor-compare = tensor_dump.compare:main",
        ]
    },
    description="Tensor dump & compare tool: txt / bin / load / compare",
    author="User",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)