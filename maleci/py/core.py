import os
from pathlib import Path
from typing import Callable, List, Union
import astroid
import astroid.arguments
import astroid.exceptions

from maleci.core import write_lines

DEFAULT_SPACES = 4
MAX_SINGLE_LINE_ARGS = 3
NO_FILE_NAMES = ["", ".", "<", ">"]


def find_nodes(tree, check, surface=False, do_not_go_into=None, root=True):

    result = []

    if check(tree):
        result.append(tree)

    if do_not_go_into is not None and not root:
        if do_not_go_into(tree):
            return result

    if surface:
        for node in tree.get_children():
            if check(node):
                result.append(node)
    else:
        for node in tree.get_children():
            result += find_nodes(node, check, do_not_go_into=do_not_go_into, root=False)

    return result


def infer_type(node, include_imports=False):

    if include_imports:
        if isinstance(node, astroid.Import):
            return node.names[0][0]
        elif isinstance(node, astroid.ImportFrom):
            result = []
            for name in node.names:
                result.append(node.modname + "." + name[0])
            return "|".join(result)

    try:
        infer = next(node.infer())
        return str(infer.pytype())
    except astroid.exceptions.InferenceError:
        return None


def get_all_types(node, suffix=""):

    result = []

    # if hasattr(node, "expr"):

    #     local_suffix = ""

    #     if hasattr(node, "attrname"):
    #         local_suffix = node.attrname + ("." if len(suffix) > 0 else "") + suffix

    #     result += get_all_classes(node.expr, local_suffix)

    node_type = next(node.infer())

    if isinstance(node_type, astroid.util.UninferableBase):
        return result

    if isinstance(node_type, astroid.Module) or isinstance(node_type, astroid.Name):
        result.append(node_type.name + ("." if len(suffix) > 0 else "") + suffix)
    elif isinstance(node_type, astroid.ClassDef):

        prefix = ""
        if isinstance(node_type.parent, astroid.Module):
            prefix = node_type.parent.name

        result.append(
            prefix
            + ("." if len(prefix) > 0 else "")
            + node_type.name
            + ("." if len(suffix) > 0 else "")
            + suffix
        )
    elif hasattr(node_type, "pytype"):
        result.append(node_type.pytype() + ("." if len(suffix) > 0 else "") + suffix)

    if hasattr(node_type, "bases"):
        for base in node_type.bases:
            result += get_all_types(base, suffix)

    return result


def get_all_unique_types(node):

    result = []

    for r in get_all_types(node):
        if not r in result:
            result.append(r)

    return result


def find_if_type_is_based_on(node, target: str):
    return target in get_all_types(node)


def find_nodes_by_type(tree, type):
    cel_nodes = find_nodes(tree, lambda x: type == infer_type(x))

    return cel_nodes


def parse_file(path, module_name=""):
    with open(path, "r") as f:
        text = f.read()

    tree = astroid.parse(text, module_name, path)
    tree.lines = text.splitlines()

    return tree


def create_trees_from_folder(folder_path: str, filter=lambda x: ".py" == x[-3:]):
    subdirs = os.walk(folder_path)

    all_files = []

    for subdir, _, files in subdirs:
        for file in files:
            if filter(file):
                all_files.append(
                    (
                        subdir.replace(folder_path, "").replace("/", "."),
                        os.path.join(subdir, file),
                        file[:-3],
                    )
                )

    all_trees = []

    for prefix, file, module_name in all_files:

        tree = parse_file(file, prefix + ("." if len(prefix) > 0 else "") + module_name)
        all_trees.append(tree)

    return all_trees


def last_import_line_number(tree):

    result = 0

    for node in tree.body:
        if isinstance(node, astroid.Import) or isinstance(node, astroid.ImportFrom):
            result = node.lineno

    return result


def insert_lines(tree, index, lines):

    if index == -1:
        index = len(tree.lines)
        tree.lines += lines
    else:
        tree.lines[index:index] = lines

    if not hasattr(tree, "line_inserts"):
        tree.line_inserts = []

    tree.line_inserts.append((index, len(lines)))


def line_number_to_line_index(tree, lineno):

    result = lineno - 1

    if not hasattr(tree, "line_inserts"):
        tree.line_inserts = []

    for id, count in tree.line_inserts:
        if id < result:
            result += count

    return result


def check_if_class(node: astroid.NodeNG):
    return isinstance(node, astroid.ClassDef)


def check_if_function(node: astroid.NodeNG):
    return isinstance(node, astroid.FunctionDef)


def check_if_return(node: astroid.NodeNG):
    return isinstance(node, astroid.Return)


def check_if_function_returns(node: astroid.FunctionDef):

    return_nodes = find_nodes(node, check_if_return, False, check_if_function)

    for r in return_nodes:
        if r.value is not None:
            return True

    return False


def create_check_if_import(name: str):

    def check_if_import(node: astroid.NodeNG):
        return (
            isinstance(node, astroid.Import) and name in [a[0] for a in node.names]
        ) or (isinstance(node, astroid.ImportFrom) and node.modname == name)

    return check_if_import


def combine_checks(checks: List[Callable]):

    def combined_check(node: astroid.NodeNG):
        for check in checks:
            if not check(node):
                return False

        return True

    return combined_check


def write_tree(tree, path):
    with open(path, "w") as f:
        f.writelines([a + os.linesep for a in tree.lines])


def add_to_requirements(requrements_to_add: List[str], file: Union[str, Path]):

    existing_requirement_lines = []

    if os.path.exists(file):
        with open(file) as f:
            existing_requirement_lines = [a.rstrip() for a in f.readlines()]

    for r in requrements_to_add:
        exists = False
        for er in existing_requirement_lines:
            if er.split("=")[0] == r:
                exists = True
                break

        if not exists:
            existing_requirement_lines.append(r)

    write_lines(existing_requirement_lines, file)


class FakeInitFunction(astroid.FunctionDef):
    def __init__(self):

        name = "__init__"
        lineno = 0
        col_offset = 0
        end_lineno = None
        end_col_offset = None
        parent = None
        super().__init__(
            name,
            lineno,
            col_offset,
            parent,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
        )

        self.args = astroid.Arguments(parent=self, vararg=None, kwarg=None)

        self.args.kwonlyargs = []
        self.args.args = [
            astroid.Name("self", 0, 0, self.args, end_lineno=0, end_col_offset=0)
        ]
