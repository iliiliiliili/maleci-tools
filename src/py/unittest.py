import astroid
import os
from pathlib import Path

from exceptions import CancelException, NotFolderException
from py.core import (
    find_nodes,
    last_import_line_number,
    insert_lines,
    line_number_to_line_index,
    parse_file,
    backup_file,
    write_tree
)

from core import find_files_in_folder, get_project_path, path_in_project, select_continue_with_details, display_files

EXPECTED_ARGS = {
    "add unittest": [("path", "folder"), ("output", "output_path", "output_folder", "out", "out_path", "out_folder", "target", "test", "tests")]
}

DEFAULT_VALUES = {
    "add unittest": {
        "path": ".",
        "output": "tests"
    }
}

FILE_SELECT_MESSAGE = "Select to which file add fire"

def verify_and_fix_args(args, project):

    if args["path"] == "":
        args["path"] == "."
    if args["output"] == "":
        args["output"] == "."

    project_path = get_project_path(project)
    args["path"] = path_in_project(args["path"], project_path)
    args["output"] = path_in_project(args["output"], project_path)

    if not os.path.isdir(args["path"]):
        raise NotFolderException()

    return args


def add_unittests_to_folder(path, output, project):

    project_path = get_project_path(project)
    files = find_files_in_folder(path)

    if not select_continue_with_details(
        f"Targeting {len(files)} files in {path} to create unit tests and save to {output}",
        details_func=lambda: display_files(files, project_path),
        details_text="See files"
    ):
        raise CancelException("")

    # tree = add_fire_to_tree(parse_file(path))
    # backup_file(path, "add fire")
    # write_tree(tree, path)
    
    
    