from setuptools import find_packages, setup

LONG_DESCRIPTION = """
Maleci tools
""".strip()

SHORT_DESCRIPTION = """
Maleci tools""".strip()

DEPENDENCIES = ["astroid", "simple-term-menu"]

TEST_DEPENDENCIES = []

VERSION = "0.1.0"
URL = "https://github.com/iliiliiliili/maleci-tools"

setup(
    name="maleci",
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    author="Illia Oleksiienko",
    author_email="io@ece.au.dk",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: The MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Operating System :: Unix",
    ],
    keywords="command line interface cli tool",
    packages=find_packages("."),
    # package_dir={"maleci": "src", "maleci.py": "src/py"},
    entry_points={
        "console_scripts": [
            "maleci = maleci.__main__:execute_script",
        ],
    },
    install_requires=DEPENDENCIES,
    tests_require=TEST_DEPENDENCIES,
)
