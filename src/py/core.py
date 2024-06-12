import os
import shutil
import astroid
import astroid.exceptions
from pathlib import Path

BACKUP_PATH = "./backup"
BACKUP_ID_FILE = os.path.join(BACKUP_PATH, ".id")


def find_nodes_old(tree, check):

    result = []

    if check(tree):
        result.append(tree)

    if hasattr(tree, "body"):
        for node in tree.body:
            result += find_nodes_old(node, check)
    elif hasattr(tree, "targets"):
        for target in tree.targets:
            result += find_nodes_old(target, check)

    return result


def find_nodes(tree, check, surface=False):

    result = []

    if check(tree):
        result.append(tree)

    if surface:
        for node in tree.get_children():
            if check(node):
                result.append(node)
    else:
        for node in tree.get_children():
            result += find_nodes(node, check)

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


def write_tree(tree, path):
    with open(path, "w") as f:
        f.writelines([a + os.linesep for a in tree.lines])


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