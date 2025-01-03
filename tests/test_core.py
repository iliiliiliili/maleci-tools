from pathlib import Path
import unittest
import shutil
import os

from maleci.core import (
    get_args,
    py_filter,
    find_code_files_in_folder,
    resolve_path,
    path_in_project,
    get_relative_path,
    display_files,
    to_camel_case,
    to_snake_case,
    insert_lines_with_indendtation,
    indent,
    indent_single,
    write_lines,
    backup_file,
)
from maleci.py.unittest import EXPECTED_ARGS, DEFAULT_VALUES

class TestCore(unittest.TestCase):

    def test_get_args(self):
        args = ["path", "target"]
        kwargs = {
            "spaces": 13, "no": True
        }
        expected_args = EXPECTED_ARGS["py add unittest"]
        default_values = DEFAULT_VALUES["py add unittest"]

        output = get_args(
            args,
            kwargs,
            expected_args,
            default_values,
        )

        self.assertDictEqual(output, {
            "path": "path",
            "output": "target",
            "spaces": 13,
            "overwrite_tests": False,
            "make_scripts": True,
            "yes": False,
        })

    def test_py_filter(self):
        filenames = ["a.py", "asd", "x.cpp", "x"]

        output = [py_filter(a) for a in filenames]

        self.assertEqual(output, [True, False, False, False])

    def test_find_files_in_folder(self):
        folder_path = "./"
        filter = lambda name: name == "run_tests.sh"

        output = find_code_files_in_folder(folder_path, filter)

        self.assertEqual(len(output), 1)

    def test_resolve_path(self):
        path = "./../."

        output = resolve_path(path)

        self.assertEqual(output, Path("..").resolve().absolute())

    def test_path_in_project(self):
        path = "py"
        project_path = Path("./tests")

        output = path_in_project(path, project_path)

        self.assertEqual(output, str(Path("./tests/py").absolute()))

    def test_get_relative_path(self):
        path = "./tests/py"
        project_path = Path("./tests")

        output = get_relative_path(path, project_path)

        self.assertEqual(output, Path("py"))

    def test_to_camel_case(self):
        snake_str = "snake_case"

        output = to_camel_case(snake_str)

        self.assertEqual(output, "SnakeCase")

    def test_to_snake_case(self):
        str = "CamelCase"

        output = to_snake_case(str)

        self.assertEqual(output, "camel_case")

    def test_insert_lines_with_indendtation(self):
        target = ["start", "end"]
        index = 1
        lines = ["l1", "l2"]
        spaces = 1
        indentation_level = 1

        insert_lines_with_indendtation(
            target,
            index,
            lines,
            spaces,
            indentation_level,
        )

        self.assertEqual(
            target,
            [
                "start",
                " l1",
                " l2",
                "end"
            ]
        )

    def test_indent(self):
        lines = ["l1", "l2"]
        spaces = 1
        indentation_level = 3

        output = indent(lines, spaces, indentation_level)

        self.assertEqual(output, ["   l1", "   l2"])

    def test_indent_single(self):
        line = "l1"
        spaces = 3
        indentation_level = 2

        output = indent_single(line, spaces, indentation_level)

        self.assertEqual(output, "      l1")

    def test_write_lines(self):
        lines = ["test test", "file"]
        dir_path = "./tests/temp"
        path = f"{dir_path}/write_lines_test.txt"

        os.makedirs(dir_path, exist_ok=True)

        write_lines(lines, path)

        with open(path, "r") as f:
            text_lines = f.readlines()

        shutil.rmtree(dir_path)

        self.assertEqual([a + os.linesep for a in lines], text_lines)
