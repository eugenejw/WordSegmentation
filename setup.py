# -*- coding: utf-8 -*-

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

import wordsegmentation

class Tox(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import tox
        errno = tox.cmdline(self.test_args)
        sys.exit(errno)

with open('README.rst') as fptr:
    readme = fptr.read()

with open('LICENSE') as fptr:
    license = fptr.read()

setup(
    name='wordsegmentation',
    version=wordsegmentation.__version__,
    description='English word segmentation.',
    long_description=readme,
    keywords='English word segmentation segment url domain word break',
    author='Weihan Jiang',
    author_email='weihan.github@gmail.com',
    license=license,
    py_modules=['wordsegmentation'],
    packages=['corpus'],
    install_requires=['networkx'],
    package_data={'corpus': ['*.*']},
    tests_require=['tox'],
    cmdclass={'test': Tox},
    platforms='any',
)
