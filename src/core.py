import os
from pathlib import Path
from typing import Union
from simple_term_menu import TerminalMenu

def get_args(args, kwargs, expected_args, default_values):

    result = {}

    for i, arg_value in enumerate(args):
        if len(expected_args) > i:

            arg_name = expected_args[i]

            if isinstance(arg_name, tuple):
                arg_name = arg_name[0]

            result[arg_name] = arg_value
    
    for arg_names in expected_args:
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
                        raise ValueError(f"No value for required argument {arg_name}")
    
    return result


def py_filter(filename):
    return ".py" == filename[-3:]


def find_files_in_folder(folder_path: str, filter=py_filter):
    subdirs = os.walk(folder_path)

    all_files = []

    for subdir, _, files in subdirs:
        for file in files:
            if filter(file):
                all_files.append(
                    (
                        subdir,
                        os.path.join(subdir, file),
                        file,
                    )
                )

    return all_files


def select_option(options, message):
    print(message)
    menu = TerminalMenu(options)
    index = menu.show()

    if index is None:
        raise ValueError("No selection")

    return index


def get_project_path(project: str):
    return Path(project).resolve()


def path_in_project(path: Union[str, Path], project_path: Path):
    return str((project_path / path).resolve())


def get_display_path(path: Union[str, Path], project_path: Path):
    return Path(path).relative_to(project_path)