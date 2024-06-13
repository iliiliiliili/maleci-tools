import unittest

from src.py.fire import (
    verify_and_fix_args,
    add_fire_to_tree,
    add_fire_to_file,
)
class TestFire(unittest.TestCase):

    def test_verify_and_fix_args(self):
        args = None
        project = None

        output = verify_and_fix_args(args, project)

        self.assertEqual(output, None)

    def test_add_fire_to_tree(self):
        tree = None

        output = add_fire_to_tree(tree)

        self.assertEqual(output, None)

    def test_add_fire_to_file(self):
        path = None

        add_fire_to_file(path)

        self.assertTrue(False)
