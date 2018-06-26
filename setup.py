from setuptools import setup, find_packages


setup(
    name='asmjit',
    version='0.1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    url='',
    license='AGPLv3',
    author='Julian Hammer',
    author_email='julian.hammer@u-sys.org',
    description='A Benchmark Toolkit for Assembly Instructions Using the LLVM JIT',
    install_requires=['llvmlite>=0.23.2', 'psutil'],
)
