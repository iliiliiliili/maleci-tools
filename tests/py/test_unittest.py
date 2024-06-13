import unittest

from src.py.unittest import (
    verify_and_fix_args,
    make_test,
    add_unittest_for_file,
    add_unittests_to_folder,
)
class TestUnittest(unittest.TestCase):

    def test_verify_and_fix_args(self):
        args = None
        project = None

        output = verify_and_fix_args(args, project)

        self.assertEqual(output, None)

    def test_make_test(self):
        tree = None
        name = None
        subdir = None
        source_folder = None
        spaces = None
        project_path = None

        output = make_test(
            tree,
            name,
            subdir,
            source_folder,
            spaces,
            project_path,
        )

        self.assertEqual(output, None)

    def test_add_unittest_for_file(self):
        file_path = None
        name = None
        subdir = None
        source_folder = None
        output_folder = None
        spaces = None
        overwrite_tests = None
        project_path = None

        add_unittest_for_file(
            file_path,
            name,
            subdir,
            source_folder,
            output_folder,
            spaces,
            overwrite_tests,
            project_path,
        )

        self.assertTrue(False)

    def test_add_unittests_to_folder(self):
        path = None
        output = None
        yes = None
        overwrite_tests = None
        spaces = None
        project = None

        add_unittests_to_folder(
            path,
            output,
            yes,
            overwrite_tests,
            spaces,
            project,
        )

        self.assertTrue(False)
