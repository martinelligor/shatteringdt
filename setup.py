#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

setup_args = dict(
    author="Igor Martinelli",
    author_email="igor.martinelli03@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    description="Useful tools to work with Statistical Learning Theory (SLT) over scikit-learn decision trees.",
    entry_points={
        "console_scripts": [
            "shatteringdt=shatteringdt.cli:main",
        ],
    },
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type="text/markdown",
    package_data={"shatteringdt": ["py.typed"]},
    include_package_data=True,
    name="shatteringdt",
    package_dir={"": "shatteringdt"},
    packages=find_packages(include=["shatteringdt/.*"]),
    setup_requires=[],
    url="https://github.com/martinelligor/shatteringdt",
    keywords = ['Machine Learning', 'Shattering Coefficient', 'Chernoff Bound', 'Decision Tree', 'Statistical Learning Theory'],
    version="0.1.0",
    zip_safe=False,
)

install_requires=[
        'scikit-learn',
        'matplotlib',
        'numpy',
        'tqdm'
    ],

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)