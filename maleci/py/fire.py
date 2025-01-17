import astroid
import os
from pathlib import Path

from maleci.py.core import (
    find_nodes,
    last_import_line_number,
    insert_lines,
    line_number_to_line_index,
    parse_file,
    write_tree,
    create_check_if_import,
    add_to_requirements,
    NO_FILE_NAMES
)

from maleci.core import (
    backup_file,
    find_code_files_in_folder,
    select_option,
    resolve_path,
    path_in_project,
    get_relative_path,
)

from maleci.exceptions import NoSelectionException, VerificationCancelledException

EXPECTED_ARGS = {"py add fire": [("path", "file"), "silent", ("requirements_path", "requirements")]}

DEFAULT_VALUES = {
    "py add fire": {
        "path": ".",
        "silent": False,
        "requirements_path": "requirements.txt",
    }
}

FILE_SELECT_MESSAGE = "Select to which file add fire"


def verify_and_fix_args(args, project):

    if args["path"] == "":
        args["path"] == "."

    project_path = resolve_path(project)
    args["path"] = path_in_project(args["path"], project_path)

    if args["requirements_path"] in NO_FILE_NAMES:
        args["requirements_path"] = None
    else:
        args["requirements_path"] = path_in_project(args["requirements_path"], project_path)

    args["path"] = path_in_project(args["path"], project_path)

    if os.path.isdir(args["path"]):
        files = find_code_files_in_folder(args["path"])

        options = [f"{a[2]} in \033[93m{get_relative_path(a[0], project_path)}\033[0m" for a in files]
        try:
            index = select_option(options, FILE_SELECT_MESSAGE)
            args["path"] = files[index][1]
        except NoSelectionException:
            raise VerificationCancelledException()

    return args


def check_if_name_main_call(node):

    check_direct = (
        isinstance(node, astroid.If)
        and isinstance(node.test, astroid.Compare)
        and isinstance(node.test.left, astroid.Name)
        and (node.test.left.name == "__name__")
        and len(node.test.ops) > 0
        and (node.test.ops[0][0] == "==")
        and isinstance(node.test.ops[0][1], astroid.Const)
        and (node.test.ops[0][1].value == "__main__")
    )

    if not check_direct:
        check_reverse = (
            isinstance(node, astroid.If)
            and isinstance(node.test, astroid.Compare)
            and isinstance(node.test.left, astroid.Const)
            and (node.test.left.value == "__main__")
            and len(node.test.ops) > 0
            and (node.test.ops[0][0] == "==")
            and isinstance(node.test.ops[0][1], astroid.Name)
            and (node.test.ops[0][1].name == "__name__")
        )

        return check_reverse

    return check_direct


def check_if_fire_call(node):

    try:
        infer = next(node.infer())
    except astroid.InferenceError:
        return False

    return (
        isinstance(infer, astroid.FunctionDef)
        and (infer.name == "Fire")
        and isinstance(infer.parent, astroid.Module)
        and infer.parent.name == "fire.core"
    )


def add_fire_to_tree(tree: astroid.Module, silent: bool):

    fire_import = "from fire import Fire".splitlines()

    fire_call = (
        lambda name="": f"""


if __name__ == "__main__":
    Fire({name})""".splitlines()
    )

    old_name_main = lambda x: f"def old_name_main{('_' + x if x > 0 else '')}():"

    check_if_import_fire = create_check_if_import("fire")

    import_fire_nodes = find_nodes(tree, check_if_import_fire, True)
    name_main_call_nodes = find_nodes(tree, check_if_name_main_call, True)

    function_nodes = find_nodes(
        tree, lambda node: isinstance(node, astroid.FunctionDef), True
    )

    has_fire = len(import_fire_nodes) > 0
    has_name_main_call = len(name_main_call_nodes) > 0
    has_fire_call = False

    if not has_fire:
        import_location = last_import_line_number(tree)
        insert_lines(tree, import_location, fire_import)
    else:
        if not silent:
            path = tree.name if len(tree.name) > 0 else tree.path
            print(f"Fire import already included into \033[93m{path}\033[0m")

    if has_name_main_call:

        for i, node in enumerate(name_main_call_nodes):

            fire_calls = find_nodes(node, check_if_fire_call)

            if len(fire_calls) > 0:
                has_fire_call = True
                if not silent:
                    path = tree.name if len(tree.name) > 0 else tree.path
                    print(f"Fire call already included into \033[93m{path}\033[0m")
            else:
                tree.lines[line_number_to_line_index(tree, node.lineno)] = (
                    old_name_main(i)
                )

    if not has_fire_call:

        fire_call_lines = (
            fire_call(function_nodes[0].name)
            if len(function_nodes) == 1
            else fire_call()
        )

        insert_lines(tree, -1, fire_call_lines)
        if not silent:
            path = tree.name if len(tree.name) > 0 else tree.path
            print(f"Added fire to \033[93m{path}\033[0m")

    return tree


def add_fire_to_file(path, silent, requirements_path, backup=True):
    tree = add_fire_to_tree(parse_file(path), silent)
    
    if backup:
        backup_file(path, "py add fire")

    write_tree(tree, path)

    if requirements_path is not None:
        add_to_requirements(["fire"], requirements_path)
