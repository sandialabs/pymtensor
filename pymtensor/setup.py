import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymtensor-pymtensor-group", # Replace with your own username
    version="1.0.0",
    author="Daniel Jensen",
    author_email="dsjense@sandia.gov",
    description="Material tensor package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danielsjensen1/pymtensor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)