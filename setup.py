from setuptools import setup, find_packages

setup(
    name="rl-laboratory",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "gymnasium",
        "tqdm",
        "numpy"
    ], # Populate universal dependecies later
    extras_require={
        "ceviche": ["ceviche"]
    }, # Populatee environment specific dependecies
    python_requies=">=3.8", # Might edit required version later.
    author="Matthew Villescas and Selin Ertan",
    description="edit this descripition later",
    long_description=open("README.md").read(),
    long_description_content_type="test/markdown",
    classifiers=[],
)