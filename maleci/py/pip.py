import astroid
import os
from pathlib import Path

from maleci.py.core import NO_FILE_NAMES, add_to_requirements

from maleci.core import (
    resolve_path,
    path_in_project,
)

EXPECTED_ARGS = {
    "py pip install": [
        None,
        ("requirements_path", "requirements", "r"),
        ("packages", "p"),
    ]
}

DEFAULT_VALUES = {
    "py pip install": {"requirements_path": "requirements.txt", "packages": []}
}


def verify_and_fix_args(args, project):

    project_path = resolve_path(project)

    if args["requirements_path"] in NO_FILE_NAMES:
        args["requirements_path"] = None
    else:
        args["requirements_path"] = path_in_project(
            args["requirements_path"], project_path
        )

    return args


def pip_install(args, requirements_path, packages):

    if isinstance(packages, str):
        packages = [packages]

    if not (isinstance(packages, list) or isinstance(packages, tuple)):
        raise ValueError("Packages should be a name or a list of names")

    args = [*args, *packages]

    if len(args) <= 0:
        raise ValueError("No packages to install")

    packages = " ".join(args)

    os.system(f"pip install {packages}")

    if requirements_path is not None:
        add_to_requirements(args, requirements_path)
