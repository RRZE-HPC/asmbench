from setuptools import setup, find_packages


with open('README.rst') as f:
    long_description = f.read()

setup(
    name='asmbench',
    version='0.1.2',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    url='https://github.com/RRZE-HPC/asmbench',
    license='AGPLv3',
    author='Julian Hammer',
    author_email='julian.hammer@fau.de',
    description='A Benchmark Toolkit for Assembly Instructions Using the LLVM JIT',
    long_description=long_description,
    install_requires=['llvmlite>=0.23.2', 'psutil'],
    extras_require={
        'sc18src': ['numpy', 'matplotlib'],
        'iaca': ['kerncraft'],
    },
    include_package_data=True,
)
