import unittest

from src.py.core import (
    find_nodes_old,
    find_nodes,
    infer_type,
    get_all_types,
    get_all_unique_types,
    find_if_type_is_based_on,
    find_nodes_by_type,
    parse_file,
    create_trees_from_folder,
    last_import_line_number,
    insert_lines,
    line_number_to_line_index,
    check_if_class,
    check_if_function,
    check_if_return,
    check_if_function_returns,
    create_check_if_import,
    combine_checks,
    write_tree,
)

from src.py.core import (
    FakeInitFunction,
)
class TestCore(unittest.TestCase):

    def test_find_nodes_old(self):
        tree = None
        check = None

        output = find_nodes_old(tree, check)

        self.assertEqual(output, None)

    def test_find_nodes(self):
        tree = None
        check = None
        surface = None

        output = find_nodes(tree, check, surface)

        self.assertEqual(output, None)

    def test_infer_type(self):
        node = None
        include_imports = None

        infer_type(node, include_imports)

        self.assertTrue(False)

    def test_get_all_types(self):
        node = None
        suffix = None

        output = get_all_types(node, suffix)

        self.assertEqual(output, None)

    def test_get_all_unique_types(self):
        node = None

        output = get_all_unique_types(node)

        self.assertEqual(output, None)

    def test_find_if_type_is_based_on(self):
        node = None
        target = None

        output = find_if_type_is_based_on(node, target)

        self.assertEqual(output, None)

    def test_find_nodes_by_type(self):
        tree = None
        type = None

        output = find_nodes_by_type(tree, type)

        self.assertEqual(output, None)

    def test_parse_file(self):
        path = None
        module_name = None

        output = parse_file(path, module_name)

        self.assertEqual(output, None)

    def test_create_trees_from_folder(self):
        folder_path = None
        filter = None

        output = create_trees_from_folder(folder_path, filter)

        self.assertEqual(output, None)

    def test_last_import_line_number(self):
        tree = None

        output = last_import_line_number(tree)

        self.assertEqual(output, None)

    def test_insert_lines(self):
        tree = None
        index = None
        lines = None

        insert_lines(tree, index, lines)

        self.assertTrue(False)

    def test_line_number_to_line_index(self):
        tree = None
        lineno = None

        output = line_number_to_line_index(tree, lineno)

        self.assertEqual(output, None)

    def test_check_if_class(self):
        node = None

        output = check_if_class(node)

        self.assertEqual(output, None)

    def test_check_if_function(self):
        node = None

        output = check_if_function(node)

        self.assertEqual(output, None)

    def test_check_if_return(self):
        node = None

        output = check_if_return(node)

        self.assertEqual(output, None)

    def test_check_if_function_returns(self):
        node = None

        output = check_if_function_returns(node)

        self.assertEqual(output, None)

    def test_create_check_if_import(self):
        name = None

        output = create_check_if_import(name)

        self.assertEqual(output, None)

    def test_combine_checks(self):
        checks = None

        output = combine_checks(checks)

        self.assertEqual(output, None)

    def test_write_tree(self):
        tree = None
        path = None

        write_tree(tree, path)

        self.assertTrue(False)


class TestFakeInitFunction(unittest.TestCase):

    def test_constructor(self):
        fake_init_function = FakeInitFunction()

        self.assertTrue(False)
