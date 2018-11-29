import os
import sys

try:
    from setuptools import setup, find_packages
except:
    from distutils.core import setup

def read_file(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='antu',
    version='0.0.1',
    author='AntNLP',
    author_email='taoji.cs@gmail.com',
    description='Universal data IO module in NLP tasks (for AntNLP Group)',
    long_description = read_file("README.txt"),
    license='Apache',
    packages=find_packages(),
    install_requires=[],
    classifiers = [
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    url ="https://github.com/AntNLP/pyAnt",
    zip_safe=True,
    include_package_data=True,
    platforms='any',
)