import unittest

from unittest.exceptions import (
    NoSelectionException,
    CancelException,
    NotFileException,
    NotFolderException,
    NoArgumentException,
    NoError,
)


class TestNoSelectionException(unittest.TestCase):

    def test_constructor(self):
        no_selection_exception = NoSelectionException()

        self.assertTrue(False)


class TestCancelException(unittest.TestCase):

    def test_constructor(self):
        cancel_exception = CancelException()

        self.assertTrue(False)


class TestNotFileException(unittest.TestCase):

    def test_constructor(self):
        not_file_exception = NotFileException()

        self.assertTrue(False)


class TestNotFolderException(unittest.TestCase):

    def test_constructor(self):
        not_folder_exception = NotFolderException()

        self.assertTrue(False)


class TestNoArgumentException(unittest.TestCase):

    def test_constructor(self):
        no_argument_exception = NoArgumentException()

        self.assertTrue(False)


class TestNoError(unittest.TestCase):

    def test_constructor(self):
        args=[]

        no_error = NoError(*args)

        self.assertTrue(False)

    def test_really_not_an_exception(self):
        args=[]

        no_error = NoError(*args)

        a = None
        b = None

        output = no_error.really_not_an_exception(a, b)

        self.assertEqual(output, None)
