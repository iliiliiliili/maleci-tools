import unittest

from maleci.exceptions import (
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

        self.assertTrue(no_selection_exception is not None)


class TestCancelException(unittest.TestCase):

    def test_constructor(self):
        cancel_exception = CancelException()

        self.assertTrue(cancel_exception is not None)


class TestNotFileException(unittest.TestCase):

    def test_constructor(self):
        not_file_exception = NotFileException()

        self.assertTrue(not_file_exception is not None)


class TestNotFolderException(unittest.TestCase):

    def test_constructor(self):
        not_folder_exception = NotFolderException()

        self.assertTrue(not_folder_exception is not None)


class TestNoArgumentException(unittest.TestCase):

    def test_constructor(self):
        no_argument_exception = NoArgumentException()

        self.assertTrue(no_argument_exception is not None)


class TestNoError(unittest.TestCase):

    def test_constructor(self):
        args=[]

        no_error = NoError(*args)

        self.assertTrue(no_error is not None)

    def test_really_not_an_exception(self):
        args=[]

        no_error = NoError(*args)

        a = ""
        b = ""

        output = no_error.really_not_an_exception(a, b)

        self.assertEqual(output, "")
