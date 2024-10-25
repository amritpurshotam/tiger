import pathlib

import pkg_resources
from setuptools import find_packages, setup


def get_requirements(requirements_path: str):
    with pathlib.Path(requirements_path).open() as requirements_txt:
        requirements = [
            str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]
    return requirements


setup(
    name="tiger",
    packages=find_packages(include=["tiger"]),
    version="1.0.0",
    description="",
    author="",
    python_requires=">=3.10",
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": get_requirements("scripts/requirements-dev.txt"),
        "notebook": get_requirements("scripts/requirements-nb.txt"),
        "test": get_requirements("scripts/requirements-test.txt"),
    },
)
