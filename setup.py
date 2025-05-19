from setuptools import setup, find_packages

setup(
    name='saer',
    version='0.1.0',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "transformers",
        "datasets",
        "torch",
        "numpy<2.0.0",
        "ipykernel",
        "ipywidgets",
        "accelerate",
        "tqdm",
        "matplotlib",
        "vllm"
    ],
)
