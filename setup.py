import re
from setuptools import setup, find_packages


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/_version.py').read())
    return result.group(1)


with open("README.md", "r") as fh:
    long_description = fh.read()

PROJECT_NAME = "histopathTDA"

setup(
    name="histopathTDA",
    version=get_property('__version__', PROJECT_NAME),
    author="Demi Qin, Jordan Schupbach",
    author_email="jordan.schupbach@montana.edu",
    description="A package for doing histopathology with TDA methods",
    keywords="tda, topology, histopathology",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/comptag/histopathTDA",
    packages=find_packages(exclude=['tests*']),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    include_package_data=True,
    package_data={'': ['_assets/example_images/*.png']},
    # TODO: Add install_requires field?
)
