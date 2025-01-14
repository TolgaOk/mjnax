import os
from setuptools import setup

dir_path = os.path.abspath(os.path.dirname(__file__))


def get_version():
    """Gets the gymnasium version."""
    path = os.path.join(dir_path, "mjnax", "__init__.py")
    lines = open(path, "r").readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    # Metadata
    name="mjnax",
    version=get_version(),
    author="Tolga Ok",
    author_email="tok@tudelft.nl",
    url="",
    description="MuJoCo environments implemented in gymnax",
    long_description=(""),
    license="MIT",

    # Package info
    packages=["mjnax", "mjnax.assets"],
    package_data={
        "mjnax": ["assets/*"],
    },
    install_requires=[],
    zip_safe=False
)
