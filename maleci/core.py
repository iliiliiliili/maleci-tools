import os
import shutil
from pathlib import Path
from typing import Union
from simple_term_menu import TerminalMenu
import fnmatch

from maleci.exceptions import NoArgumentException, NoSelectionException


BACKUP_PATH = Path.home() / ".maleci-backup"
BACKUP_ID_FILE = os.path.join(BACKUP_PATH, ".id")


def get_args(args, kwargs, expected_args, default_values):

    result = {}

    if expected_args[0] is None:

        result = {"args": args}
    else:
        for i, arg_value in enumerate(args):

            if len(expected_args) > i:

                arg_name = expected_args[i]

                if isinstance(arg_name, tuple):
                    arg_name = arg_name[0]

                result[arg_name] = arg_value

    for arg_names in expected_args:

        if arg_names is None:
            continue

        if isinstance(arg_names, tuple):
            arg_name = arg_names[0]
        else:
            arg_name = arg_names

        if not (arg_name in result):
            if arg_name in kwargs:
                result[arg_name] = kwargs[arg_name]
            else:

                found = False

                if isinstance(arg_names, tuple):
                    for name in arg_names:
                        if name in kwargs:
                            result[arg_name] = kwargs[name]
                            found = True
                            break

                if not found:

                    if arg_name in default_values:
                        result[arg_name] = default_values[arg_name]
                    else:
                        raise NoArgumentException(
                            f"No value for required argument {arg_name}"
                        )

    return result


def py_filter(filename):
    return ".py" == filename[-3:]


def read_gitignore(project_path: Path):
    """Read .gitignore file and return list of patterns"""
    gitignore_file = project_path / ".gitignore"
    if not gitignore_file.exists():
        return []

    with open(gitignore_file) as f:
        patterns = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]
    return patterns


def matches_gitignore(path: str, base_path: Path, patterns: list) -> bool:
    """Check if path matches any gitignore pattern"""
    rel_path = Path(path).relative_to(base_path)
    return any(
        (
            fnmatch.fnmatch(str(rel_path), pattern)
            or (
                len(rel_path.parts) > 0
                and fnmatch.fnmatch(str(rel_path.parts[0]), pattern)
            )
        )
        for pattern in patterns
    )


def find_code_files_in_folder(folder: str, filter=py_filter, excluded_folders=None):
    subdirs = os.walk(folder)
    base_path = Path(folder)
    all_files = []

    # Read gitignore patterns once
    gitignore_patterns = read_gitignore(base_path)

    for subdir, _, files in subdirs:
        # Skip if this is a top-level excluded folder
        if excluded_folders:
            rel_path = Path(subdir).relative_to(base_path)
            if len(rel_path.parts) > 0 and rel_path.parts[0] in excluded_folders:
                continue

        # Skip if directory matches gitignore
        if gitignore_patterns and matches_gitignore(
            subdir, base_path, gitignore_patterns
        ):
            continue

        for file in files:
            file_path = os.path.join(subdir, file)
            if filter(file):

                # Skip if file matches gitignore
                if gitignore_patterns and matches_gitignore(
                    file_path, base_path, gitignore_patterns
                ):
                    continue

                all_files.append(
                    (
                        subdir,
                        file_path,
                        file,
                    )
                )

    return all_files


def select_option(options, message, show_selected_option=True):
    print(message)
    menu = TerminalMenu(options)

    try:
        index = menu.show()
    except KeyboardInterrupt:
        print("Exiting")
        exit(0)

    if index is None:
        raise NoSelectionException("")

    if show_selected_option:
        print(f"\033[92m    {options[index]}\033[0m")

    return index


def select_continue(message):

    index = select_option(["Continue", "Cancel"], message)

    return index == 0


def select_continue_with_details(message, details_func, details_text="Details"):

    index = select_option(["Continue", details_text, "Cancel"], message)

    if index == 1:
        details_func()
        return select_continue(message)

    return index == 0


def resolve_path(path: str):
    return Path(path).expanduser().resolve()


def path_in_project(path: Union[str, Path], project_path: Path):
    return str((project_path / path).resolve())


def get_relative_path(path: Union[str, Path], project_path: Path):
    return Path(path).relative_to(project_path)


def display_files(files, project_path: Path):
    for file in files:
        print(f"{file[2]} in \033[93m{get_relative_path(file[0], project_path)}\033[0m")


def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_snake_case(str):
    return "".join(["_" + i.lower() if i.isupper() else i for i in str]).lstrip("_")


def insert_lines_with_indendtation(target, index, lines, spaces, indentation_level=1):

    if index == -1:
        target += indent(lines, spaces, indentation_level)
    else:
        target[index:index] = indent(lines, spaces, indentation_level)


def indent(lines, spaces, indentation_level=1):
    return [indent_single(a, spaces, indentation_level) for a in lines]


def indent_single(line, spaces, indentation_level=1):
    return line if line == "" else (spaces * indentation_level * " " + line)


def write_lines(lines, path):
    with open(path, "w") as f:
        f.writelines([a + os.linesep for a in lines])


def create_file_from_code(code: str, path: str, project: str):
    full_path = (resolve_path(project) / path).resolve()
    full_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_path, "w") as f:
        f.write(code)


def backup_file(file_path, action):
    # TODO

    if not os.path.exists(BACKUP_PATH):
        os.makedirs(BACKUP_PATH, exist_ok=True)

    if not os.path.exists(BACKUP_ID_FILE):
        with open(BACKUP_ID_FILE, "w") as f:
            pass

    with open(BACKUP_ID_FILE, "r") as f:
        ids = [(int(a[0]), a[1]) for a in [x.split("!!!", 2) for x in f.readlines()]]

    new_id = (ids[-1][0] + 1) if len(ids) > 0 else 0

    shutil.copyfile(file_path, os.path.join(BACKUP_PATH, f"{new_id}"))

    with open(BACKUP_ID_FILE, "a") as f:
        f.write(f"{new_id}!!!{action}!!!{Path(file_path).absolute()}" + os.linesep)
