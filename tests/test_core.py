import unittest

from src.core import (
    get_args,
    py_filter,
    find_files_in_folder,
    select_option,
    select_continue,
    select_continue_with_details,
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
class TestCore(unittest.TestCase):

    def test_get_args(self):
        args = None
        kwargs = None
        expected_args = None
        default_values = None

        output = get_args(
            args,
            kwargs,
            expected_args,
            default_values,
        )

        self.assertEqual(output, None)

    def test_py_filter(self):
        filename = None

        output = py_filter(filename)

        self.assertEqual(output, None)

    def test_find_files_in_folder(self):
        folder_path = None
        filter = None

        output = find_files_in_folder(folder_path, filter)

        self.assertEqual(output, None)

    def test_select_option(self):
        options = None
        message = None

        output = select_option(options, message)

        self.assertEqual(output, None)

    def test_select_continue(self):
        message = None

        output = select_continue(message)

        self.assertEqual(output, None)

    def test_select_continue_with_details(self):
        message = None
        details_func = None
        details_text = None

        output = select_continue_with_details(message, details_func, details_text)

        self.assertEqual(output, None)

    def test_resolve_path(self):
        path = None

        output = resolve_path(path)

        self.assertEqual(output, None)

    def test_path_in_project(self):
        path = None
        project_path = None

        output = path_in_project(path, project_path)

        self.assertEqual(output, None)

    def test_get_relative_path(self):
        path = None
        project_path = None

        output = get_relative_path(path, project_path)

        self.assertEqual(output, None)

    def test_display_files(self):
        files = None
        project_path = None

        display_files(files, project_path)

        self.assertTrue(False)

    def test_to_camel_case(self):
        snake_str = None

        output = to_camel_case(snake_str)

        self.assertEqual(output, None)

    def test_to_snake_case(self):
        str = None

        output = to_snake_case(str)

        self.assertEqual(output, None)

    def test_insert_lines_with_indendtation(self):
        target = None
        index = None
        lines = None
        spaces = None
        indentation_level = None

        insert_lines_with_indendtation(
            target,
            index,
            lines,
            spaces,
            indentation_level,
        )

        self.assertTrue(False)

    def test_indent(self):
        lines = None
        spaces = None
        indentation_level = None

        output = indent(lines, spaces, indentation_level)

        self.assertEqual(output, None)

    def test_indent_single(self):
        line = None
        spaces = None
        indentation_level = None

        output = indent_single(line, spaces, indentation_level)

        self.assertEqual(output, None)

    def test_write_lines(self):
        lines = None
        path = None

        write_lines(lines, path)

        self.assertTrue(False)

    def test_backup_file(self):
        file_path = None
        action = None

        backup_file(file_path, action)

        self.assertTrue(False)
