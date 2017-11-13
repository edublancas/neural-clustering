# -*- encoding: utf-8 -*-
import io
import re
import ast
from glob import glob
import os
from os.path import basename
from os.path import splitext

from setuptools import find_packages, setup

NAME = 'neural-clustering'
DESCRIPTION = 'Clustering from multielectrode array readings using Edward'
URL = 'https://github.com/edublancas/neural-noise'
EMAIL = 'e.blancas@columbia.edu'
AUTHOR = 'Eduardo Blancas'
LICENSE = 'GPL3'

REQUIRED = [
]

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('src/neural_clustering/__init__.py', 'rb') as f:
    VERSION = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    url=URL,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    install_requires=REQUIRED,
    entry_points={
        'console_scripts': ['run_yass=neural_noise.command_line'
                            ':run_yass'],
    },
)
