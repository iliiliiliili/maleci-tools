import unittest

from unittest.fire import (
    verify_and_fix_args,
    check_if_name_main_call,
    check_if_fire_call,
    add_fire_to_tree,
    add_fire_to_file,
)
class TestFire(unittest.TestCase):

    def test_verify_and_fix_args(self):
        args = None
        project = None

        output = verify_and_fix_args(args, project)

        self.assertEqual(output, None)

    def test_check_if_name_main_call(self):
        node = None

        output = check_if_name_main_call(node)

        self.assertEqual(output, None)

    def test_check_if_fire_call(self):
        node = None

        output = check_if_fire_call(node)

        self.assertEqual(output, None)

    def test_add_fire_to_tree(self):
        tree = None
        silent = None

        output = add_fire_to_tree(tree, silent)

        self.assertEqual(output, None)

    def test_add_fire_to_file(self):
        path = None
        silent = None

        add_fire_to_file(path, silent)

        self.assertTrue(False)
