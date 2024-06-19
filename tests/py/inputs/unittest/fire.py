import astroid
import os
from pathlib import Path

from py.core import (
    find_nodes,
    last_import_line_number,
    insert_lines,
    line_number_to_line_index,
    parse_file,
    write_tree,
    create_check_if_import,
)

from maleci.core import (
    backup_file,
    find_files_in_folder,
    select_option,
    resolve_path,
    path_in_project,
    get_relative_path,
)

EXPECTED_ARGS = {"py add fire": [("path", "file"), "silent"]}

DEFAULT_VALUES = {
    "py add fire": {
        "path": ".",
        "silent": False,
    }
}

FILE_SELECT_MESSAGE = "Select to which file add fire"


def verify_and_fix_args(args, project):

    if args["path"] == "":
        args["path"] == "."

    project_path = resolve_path(project)
    args["path"] = path_in_project(args["path"], project_path)

    if os.path.isdir(args["path"]):
        files = find_files_in_folder(args["path"])

        options = [f"{a[2]} in {get_relative_path(a[0], project_path)}" for a in files]

        index = select_option(options, FILE_SELECT_MESSAGE)

        args["path"] = files[index][1]

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
            print(
                f"Fire import already included into {tree.name if len(tree.name) > 0 else tree.path}"
            )

    if has_name_main_call:

        for i, node in enumerate(name_main_call_nodes):

            fire_calls = find_nodes(node, check_if_fire_call)

            if len(fire_calls) > 0:
                has_fire_call = True
                if not silent:
                    print(
                        f"Fire call already included into {tree.name if len(tree.name) > 0 else tree.path} "
                    )
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
            print(f"Added fire to {tree.name if len(tree.name) > 0 else tree.path}")

    return tree


def add_fire_to_file(path, silent):
    tree = add_fire_to_tree(parse_file(path), silent)
    backup_file(path, "py add fire")
    write_tree(tree, path)
