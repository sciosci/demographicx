import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="demographicx",
    version="0.0.1",
    author="Lizhen Liang",
    long_description=long_description,
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy', 'torch', 'scipy', 'transformers']
)
