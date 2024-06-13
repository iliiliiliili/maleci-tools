from typing import List
import astroid
import os
from pathlib import Path

from exceptions import CancelException, NotFolderException
from py.core import (
    MAX_SINGLE_LINE_ARGS,
    FakeInitFunction,
    check_if_function_returns,
    find_nodes,
    parse_file,
    check_if_class,
    check_if_function,
    DEFAULT_SPACES,
)

from core import (
    find_files_in_folder,
    indent_single,
    resolve_path,
    path_in_project,
    select_continue_with_details,
    display_files,
    get_relative_path,
    insert_lines_with_indendtation,
    to_camel_case,
    to_snake_case,
    write_lines,
    backup_file,
)

EXPECTED_ARGS = {
    "add unittest": [
        ("path", "folder", "source", "sources", "source_path", "sources_path"),
        (
            "output",
            "output_path",
            "output_folder",
            "out",
            "out_path",
            "out_folder",
            "target",
            "test",
            "tests",
        ),
        ("yes", "y", "continue", "agree"),
        ("spaces", "indentation", "indentation_size", "indentation_spaces"),
        ("overwrite_tests", "rewrite_tests", "overwrite", "rewrite"),
    ]
}

DEFAULT_VALUES = {
    "add unittest": {
        "path": ".",
        "output": "tests",
        "yes": False,
        "spaces": DEFAULT_SPACES,
        "overwrite_tests": False,
    }
}

DO_NOT_ADD_TEST_TO_FILE = [
    "__init__.py"
]

REPLACE_FUNCTION_NAMES = {
    # original function: (test name, call string, True if needs object to be created first, True if uses arguments)
    "__init__": ("constructor", lambda class_name: class_name, False, True),
    "__new__": ("new_constructor", lambda class_name: class_name, False, True),
    "__call__": ("call", lambda object_name: object_name, True, True),
    "__str__": ("str", lambda object_name: f"str({object_name})", True, False),
    "__repr__": ("repr", lambda object_name: f"repr({object_name})", True, False),
    "__len__": ("len", lambda object_name: f"len({object_name})", True, False),
    "__add__": ("add", lambda object_name: f"None + {object_name}", True, False),
    "__sub__": ("sub", lambda object_name: f"None - {object_name}", True, False),
    "__mul__": ("mul", lambda object_name: f"None * {object_name}", True, False),
    "__truediv__": (
        "truediv",
        lambda object_name: f"None / {object_name}",
        True,
        False,
    ),
    "__floordiv__": (
        "floordiv",
        lambda object_name: f"None // {object_name}",
        True,
        False,
    ),
    "__mod__": ("mod", lambda object_name: f"None % {object_name}", True, False),
    "__pow__": ("pow", lambda object_name: f"None ** {object_name}", True, False),
    "__iter__": ("iter", lambda object_name: f"[*{object_name}]", True, False),
    "__next__": ("next", lambda object_name: f"next({object_name})", True, False),
    "__getitem__": (
        "get_item",
        lambda object_name: f"{object_name}[None]",
        True,
        False,
    ),
    "__contains__": (
        "contains",
        lambda object_name: f"None in {object_name}",
        True,
        False,
    ),
    "__reversed__": (
        "reversed",
        lambda object_name: f"reversed({object_name})",
        True,
        False,
    ),
}

# TODO entry and exit


def verify_and_fix_args(args, project):

    if args["path"] == "":
        args["path"] == "."
    if args["output"] == "":
        args["output"] == "."

    project_path = resolve_path(project)
    args["path"] = path_in_project(args["path"], project_path)
    args["output"] = path_in_project(args["output"], project_path)

    if not os.path.isdir(args["path"]):
        raise NotFolderException()

    return args


def make_test(tree: astroid.Module, name, subdir, source_folder, spaces, project_path):

    name = name.replace(".py", "")

    module_path = str(get_relative_path(subdir, project_path))
    if module_path == ".":
        module_path = ""
    module_path = module_path.replace(os.sep, ".")
    module_path = ".".join([module_path, name])

    global_import_lines = ["import unittest"]
    local_import_lines = []
    test_class_lines = lambda name: [f"class Test{name}(unittest.TestCase):"]

    classes = find_nodes(tree, check_if_class, True)
    functions = find_nodes(tree, check_if_function, True)

    def create_object_name(class_name: str):
        return to_snake_case(class_name)

    def create_function_call_test_lines(function: astroid.FunctionDef, class_name=None):

        name = function.name

        is_creator = name == "__init__"

        if name in REPLACE_FUNCTION_NAMES:
            name, create_call_str, needs_object, uses_args = REPLACE_FUNCTION_NAMES[
                name
            ]
        else:
            create_call_str = (
                lambda class_name: f"{'' if class_name is None else class_name + '.'}{name}"
            )
            needs_object = class_name is not None
            uses_args = True

        if needs_object:
            class_name = create_object_name(class_name)

        has_varargs = function.args.vararg is not None
        has_kwargs = function.args.kwarg is not None
        returns_value = is_creator or check_if_function_returns(function)

        args = [a.name for a in function.args.args]

        if class_name is not None and (is_creator or function.type != "staticmethod"):
            args = args[1:]

        kwargs = [a.name for a in function.args.kwonlyargs]

        arg_lines = []

        if uses_args:

            for arg in args:
                arg_lines.append(f"{arg} = None")

            for kwarg in kwargs:
                arg_lines.append(f"{kwarg} = None")

            if has_varargs:
                arg_lines.append("args=[]")

            if has_kwargs:
                arg_lines.append("kwargs={}")

        call_lines = []

        all_values = [
            *args,
            *(["*args"] if has_varargs else []),
            *[f"{a}={a}" for a in kwargs],
            *(["**kwargs"] if has_kwargs else []),
        ]

        single_line_call = len(all_values) <= MAX_SINGLE_LINE_ARGS

        make_output_variable_name = (
            lambda index=0: f"output{'' if index == 0 else index}"
        )

        if is_creator:
            output_variable_name = create_object_name(class_name)
        else:
            output_variable_name_index = 0
            output_variable_name = make_output_variable_name(output_variable_name_index)

            while (output_variable_name in args) or (output_variable_name in kwargs):
                output_variable_name_index += 1
                output_variable_name = make_output_variable_name(
                    output_variable_name_index
                )

        if uses_args:
            if single_line_call:
                all_arg_values = ", ".join(all_values)
                if returns_value:
                    call_lines.append(
                        f"{output_variable_name} = {create_call_str(class_name)}({all_arg_values})"
                    )
                else:
                    call_lines.append(
                        f"{create_call_str(class_name)}({all_arg_values})"
                    )
            else:

                if returns_value:
                    call_lines.append(
                        f"{output_variable_name} = {create_call_str(class_name)}("
                    )
                else:
                    call_lines.append(f"{create_call_str(class_name)}(")

                for arg_value in all_values:
                    call_lines.append(indent_single(f"{arg_value},", spaces))

                call_lines.append(f")")
        else:
            if returns_value:
                call_lines.append(
                    f"{output_variable_name} = {create_call_str(class_name)}"
                )
            else:
                call_lines.append(f"{create_call_str(class_name)}")

        return (
            name,
            arg_lines,
            call_lines,
            needs_object,
            output_variable_name,
            (returns_value and not is_creator),
        )

    def create_function_test_lines(
        function: astroid.FunctionDef,
        init_function: astroid.FunctionDef = None,
        class_name=None,
    ):

        (
            name,
            arg_lines,
            call_lines,
            needs_object,
            output_variable_name,
            returns_value,
        ) = create_function_call_test_lines(function, class_name)

        test_name = f"test_{name}"

        assert_lines = []

        if returns_value:
            assert_lines.append(f"self.assertEqual({output_variable_name}, None)")
        else:
            assert_lines.append(f"self.assertTrue(False)")

        test_lines = [
            f"def {test_name}(self):",
        ]

        if needs_object:

            if class_name is None:
                raise ValueError("Class_name is not provided when object is needed")

            _, ctor_arg_lines, ctor_call_lines, _, _, _ = (
                create_function_call_test_lines(init_function, class_name)
            )
            insert_lines_with_indendtation(test_lines, -1, ctor_arg_lines, spaces)
            if len(ctor_arg_lines) > 0:
                test_lines.append("")
            insert_lines_with_indendtation(test_lines, -1, ctor_call_lines, spaces)
            test_lines.append("")

        insert_lines_with_indendtation(test_lines, -1, arg_lines, spaces)
        if len(arg_lines) > 0:
            test_lines.append("")
        insert_lines_with_indendtation(test_lines, -1, call_lines, spaces)
        test_lines.append("")
        insert_lines_with_indendtation(test_lines, -1, assert_lines, spaces)

        return test_lines

    def create_class_test_lines(name: str, functions: List[astroid.FunctionDef]):
        test_class_name = f"Test{name}"

        test_lines = [
            f"class {test_class_name}(unittest.TestCase):",
        ]

        init_function_list = [*filter(lambda a: a.name == "__init__", functions)]

        if len(init_function_list) > 0:
            init_function = init_function_list[0]
        else:
            init_function = FakeInitFunction()

        if len(functions) <= 0:
            functions = [init_function]

        for function in functions:
            if (
                function.name in REPLACE_FUNCTION_NAMES
                or len(function.name) <= 2
                or function.name[:2] != "__"
            ):
                function_lines = create_function_test_lines(
                    function, init_function, name
                )
                test_lines.append("")
                insert_lines_with_indendtation(test_lines, -1, function_lines, spaces)

        return test_lines

    module_test_lines = []

    if len(functions) > 0:

        local_import_lines.append(f"from {module_path} import (")

        module_test_lines.append(
            f"class Test{to_camel_case(name)}(unittest.TestCase):",
        )

        for function in functions:
            function_lines = create_function_test_lines(function)
            module_test_lines.append("")
            insert_lines_with_indendtation(
                module_test_lines, -1, function_lines, spaces
            )

            local_import_lines.append(indent_single(f"{function.name},", spaces))

        local_import_lines.append(f")")
        local_import_lines.append("")

    if len(classes) > 0:

        local_import_lines.append(f"from {module_path} import (")

        for cls in classes:
            class_functions = find_nodes(cls, check_if_function, True)
            class_lines = create_class_test_lines(cls.name, class_functions)
            module_test_lines.append("")
            module_test_lines.append("")
            module_test_lines += class_lines

            local_import_lines.append(indent_single(f"{cls.name},", spaces))

        local_import_lines.append(f")")
        local_import_lines.append("")

    result = [*global_import_lines, "", *local_import_lines[:-1], *module_test_lines]

    return result


def add_unittest_for_file(
    file_path,
    name,
    subdir,
    source_folder,
    output_folder,
    spaces,
    overwrite_tests,
    project_path,
):
    source_folder_path = resolve_path(source_folder)

    tree = parse_file(file_path)
    test_lines = make_test(tree, name, subdir, source_folder, spaces, project_path)

    output_path = Path(output_folder)

    if len(test_lines) > 0:

        test_folder_path = output_path / get_relative_path(subdir, source_folder_path)
        file_path = str(test_folder_path / ("test_" + name))

        os.makedirs(str(test_folder_path), exist_ok=True)

        if os.path.exists(file_path) and not overwrite_tests:
            print(
                f"Test file already exists for {name} in {get_relative_path(subdir, project_path)}"
            )
        else:
            if os.path.exists(file_path):
                backup_file(file_path, "overwrite test")

            write_lines(test_lines, file_path)

            print(
                f"Created tests for {name} in {get_relative_path(subdir, project_path)}"
            )
    else:
        print(
            f"Nothing to test for {name} in {get_relative_path(subdir, project_path)}"
        )


def add_unittests_to_folder(path, output, yes, overwrite_tests, spaces, project):

    project_path = resolve_path(project)
    files = find_files_in_folder(path)

    if not yes:
        if not select_continue_with_details(
            f"Targeting {len(files)} files in {path} to create unit tests and save them to {output}",
            details_func=lambda: display_files(files, project_path),
            details_text="See files",
        ):
            raise CancelException("")

    for subdir, file_path, name in files[1:]:

        if name in DO_NOT_ADD_TEST_TO_FILE:
            continue

        add_unittest_for_file(
            file_path, name, subdir, path, output, spaces, overwrite_tests, project_path
        )

    test_script_lines = [
        f"PYTHONPATH={get_relative_path(path, project_path)} python -m unittest discover -s {get_relative_path(output, project_path)}"
    ]

    test_script_file = str(project_path / "run_tests.sh")

    if os.path.exists(test_script_file):
        print("Test script already exists")
    else:
        write_lines(test_script_lines, test_script_file)
