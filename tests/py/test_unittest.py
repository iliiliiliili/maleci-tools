import os
from pathlib import Path
import shutil
import unittest

import astroid

from maleci.py.unittest import (
    verify_and_fix_args,
    make_test,
    add_unittest_for_file,
    add_unittests_to_folder,
    EXPECTED_ARGS,
    DEFAULT_VALUES,
)
from maleci.py.core import DEFAULT_SPACES

from maleci.core import (
    find_files_in_folder,
    get_args,
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


class TestUnittest(unittest.TestCase):

    def test_verify_and_fix_args(self):
        args = get_args(
            [],
            {"source": "unittest", "output": "tmp"},
            EXPECTED_ARGS["add unittest"],
            DEFAULT_VALUES["add unittest"],
        )
        project = "./tests/py/inputs"

        output = verify_and_fix_args(args, project)

        self.assertEqual(
            output,
            {
                "path": str(Path("./tests/py/inputs/unittest").absolute()),
                "output": str(Path("./tests/py/inputs/tmp").absolute()),
                "yes": False,
                "spaces": DEFAULT_SPACES,
                "overwrite_tests": False,
                "make_scripts": True,
            },
        )

    def test_make_test(self):
        code = """
            def f():
                pass
            
            def x():
                return

            def y():
                return 1
            
            class z():
                pass
                
            a = 10
        """

        tree = astroid.parse(code, "tmp")

        name = "tmp.py"
        subdir = "."
        spaces = DEFAULT_SPACES
        project_path = "."

        output = make_test(
            tree,
            name,
            subdir,
            spaces,
            project_path,
        )

        output = os.linesep.join(output)
        output_tree = astroid.parse(output)

        self.assertEqual(len(output_tree.body), 3 + 2)
        self.assertEqual(len(output_tree.body[-2].body), 3)
        self.assertEqual(len(output_tree.body[-1].body), 1)

    def test_add_unittest_for_file(self):
        
        project_path = resolve_path("./tests/py/inputs")
        output_folder = resolve_path(f"{project_path}/temp")
        source_folder = resolve_path(f"{project_path}/unittest")
        expected_folder = resolve_path("./tests/py/expected/unittest")
        files = find_files_in_folder(source_folder)
        overwrite_tests = False
        spaces = DEFAULT_SPACES

        self.assertTrue(len(files) > 0)

        for subdir, file_path, name in files:

            test_file_path = str(add_unittest_for_file(
                file_path,
                name,
                subdir,
                source_folder,
                output_folder,
                spaces,
                overwrite_tests,
                project_path,
            ))

            expected_file_path = str(expected_folder / get_relative_path(test_file_path, output_folder))

            with open(test_file_path, "r") as f:
                test_source_text = f.read()

            with open(expected_file_path, "r") as f:
                expected_source_text = f.read()

            self.assertEqual(test_source_text, expected_source_text)
        
        shutil.rmtree(output_folder)

    def test_add_unittests_to_folder(self):

        project_path = resolve_path("./tests/py/inputs")
        output_folder = resolve_path(f"{project_path}/temp")
        source_folder = resolve_path(f"{project_path}/unittest")
        expected_folder = resolve_path("./tests/py/expected/unittest")
        files = find_files_in_folder(source_folder)
        overwrite_tests = False
        spaces = DEFAULT_SPACES

        self.assertTrue(len(files) > 0)

        add_unittests_to_folder(
            source_folder,
            output_folder,
            True,
            overwrite_tests,
            spaces,
            False,
            project_path,
        )

        for subdir, _, name in files:
            
            test_folder_path = output_folder / get_relative_path(subdir, source_folder)
            test_file_path = str(test_folder_path / ("test_" + name))

            expected_file_path = str(expected_folder / get_relative_path(test_file_path, output_folder))

            with open(test_file_path, "r") as f:
                test_source_text = f.read()

            with open(expected_file_path, "r") as f:
                expected_source_text = f.read()

            self.assertEqual(test_source_text, expected_source_text)
        
        shutil.rmtree(output_folder)
