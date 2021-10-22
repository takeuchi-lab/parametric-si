# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='parametric-si',
    version='1.0.0',
    description='tools for parametric selective inference',
    long_description=readme,
    author='Daiki Miwa,Kazuya Sugiyama,Vo Nguyen Le Duy,Takeuchi Ichiro',
    author_email='miwa.daiki.mllab.nit@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        "numpy",
        "scipy",
        "portion",
        "sklearn"
        ]
)