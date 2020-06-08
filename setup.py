import io
import os
import re
from codecs import open  # To use a consistent encoding

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


# Stolen from pip
def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


# Stolen from pip
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Get the long description from the relevant file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='asmbench',
    version=find_version('asmbench', '__init__.py'),
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
    entry_points={
        'console_scripts': [
            'asmbench=asmbench.__main__:main'
        ]
    }
)
