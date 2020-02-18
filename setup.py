"""Setup for the mktanalytics package."""

import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Payton Ong",
    author_email="payton.ong@gmail.com",
    name='mktanalytics',
    license="MIT",
    description='mktanalytics is a python package for analyzing markets.',
    version='v0.0.2',
    long_description=README,
    url='https://github.com/paytonong/mktanalytics',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=['numpy', 'pandas'],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
