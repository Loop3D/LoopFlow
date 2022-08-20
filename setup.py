import os
import sys
import setuptools
import subprocess
from LoopFlow import __version__

head, tail = os.path.split(sys.argv[0])

try:
    cmd = 'bash .git.sh'
    subprocess.run(
        cmd.split())
    cmd = 'git tag -a {0} -m {0}'.format(
        __version__)
    subprocess.run(
        cmd.split())
except Exception as e:
    print(e)

def get_description():
    long_description = ""
    readme_file = os.path.join(head, "README.md")
    with open(readme_file, "r") as fh:
        long_description = fh.read()
    return long_description


setuptools.setup(
    name="LoopFlow",
    version=__version__,
    author="The Loop Organisation",
    author_email="mark.jessell@uwa.edu.au",
    description="Calculate Graphs and Distances from Loop Models",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/Loop3D/LoopFlow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
