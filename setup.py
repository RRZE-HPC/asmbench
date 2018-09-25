from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()

setup(
    name='asmbench',
    version='0.1.0',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    url='',
    license='AGPLv3',
    author='Julian Hammer',
    author_email='julian.hammer@fau.de',
    description='A Benchmark Toolkit for Assembly Instructions Using the LLVM JIT',
    long_description_content_type='text/markdown',
    long_description=long_description,
    install_requires=['llvmlite>=0.23.2', 'psutil'],
    extras_require={
        'sc18src': ['numpy', 'matplotlib'],
        'iaca': ['kerncraft'],
    }
)
