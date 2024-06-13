import unittest

from src.main import (
    no_func,
    select_comand,
    add,
)
class TestMain(unittest.TestCase):

    def test_no_func(self):
        g = None

        no_func(g)

        self.assertTrue(False)

    def test_select_comand(self):
        group = None

        output = select_comand(group)

        self.assertEqual(output, None)

    def test_add(self):
        command = None
        project = None
        args=[]
        kwargs={}

        add(
            command,
            *args,
            project=project,
            **kwargs,
        )

        self.assertTrue(False)
