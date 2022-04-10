import sys
import setuptools


if sys.version_info < (3,):
    raise Exception("Python 2 is not supported.")


with open("README.md", "r") as fh:
    long_description = fh.read()


# def req_file(filename):
#     with open(filename) as f:
#         content = f.readlines()
#     return [x.strip() for x in content]


# install_requires = req_file("requirements.txt")

setuptools.setup(
    name="oneflow_dlrm",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.0.1",
    description="OneFlow DLRM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    # url=__url__,
    # author=__contact_names__,
    # maintainer=__contact_names__,
    # The licence under which the project is released
    # license=__license__,
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Developers",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    # install_requires=install_requires,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    # keywords=__keywords__,
)
