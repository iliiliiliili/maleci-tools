import os
import shutil
import unittest

import astroid

from src.py.core import (
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

    def test_find_nodes(self):
        code = """
            def f():
                pass
            
            def x():
                return

            def y():
                return 1
            
            class z():
                def x():
                    pass
                
            a = 10
        """

        tree = astroid.parse(code)
        check = check_if_function
        surface = False

        output = find_nodes(tree, check, surface)

        self.assertEqual(len(output), 4)

    def test_infer_type(self):
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

        tree = astroid.parse(code)

        output = infer_type(tree.body[0])

        self.assertEqual(output, "builtins.function")

    def test_get_all_types(self):
        code = """
            class f():
                pass
            
            class x(f):
                pass

            class y(x):
                pass
            
            class z(y, f):
                pass
        """

        tree = astroid.parse(code, "tmp")

        output = get_all_types(tree.body[-1])

        self.assertEqual(output, ["tmp.z", "tmp.y", "tmp.x", "tmp.f", "tmp.f"])

    def test_get_all_unique_types(self):
        code = """
            class f():
                pass
            
            class x(f):
                pass

            class y(x):
                pass
            
            class z(y, f):
                pass
        """

        tree = astroid.parse(code, "tmp")

        output = get_all_unique_types(tree.body[-1])

        self.assertEqual(output, ["tmp.z", "tmp.y", "tmp.x", "tmp.f"])

    def test_find_if_type_is_based_on(self):
        code = """
            class f():
                pass
            
            class x(f):
                pass

            class y(x):
                pass
            
            class z(y, f):
                pass
        """

        tree = astroid.parse(code, "tmp")

        self.assertEqual(find_if_type_is_based_on(tree.body[-1], "tmp.f"), True)
        self.assertEqual(find_if_type_is_based_on(tree.body[-2], "tmp.z"), False)

    def test_find_nodes_by_type(self):
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

        tree = astroid.parse(code)

        output = find_nodes_by_type(tree, "builtins.function")

        self.assertEqual(len(output), 3)

    def test_parse_file(self):
        dir_path = "./tests/temp"
        path = f"{dir_path}/parse_file_test.py"
        module_name = "tmp"

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

        os.makedirs(dir_path, exist_ok=True)

        with open(path, "w") as f:
            f.write(code)

        output = parse_file(path, module_name)

        shutil.rmtree(dir_path)

        self.assertEqual(output.name, module_name)
        self.assertEqual(len(output.body), 5)

    def test_create_trees_from_folder(self):
        folder_path = "./src"

        output = create_trees_from_folder(folder_path)

        self.assertGreaterEqual(len(output), 7)

    def test_last_import_line_number(self):
        code = """
            import none

            def x():
                return

            import done

            def y():
                return 1
            
            class z():
                pass
                
            a = 10
        """

        tree = astroid.parse(code)
        tree.lines = code.splitlines()

        output = last_import_line_number(tree)

        self.assertEqual(output, 7)

    def test_insert_lines(self):
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

        tree = astroid.parse(code)
        tree.lines = code.splitlines()
        index = 0
        lines = ["import none"]

        insert_lines(tree, index, lines)

        self.assertEqual(tree.lines[0], "import none")

    def test_line_number_to_line_index(self):
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

        tree = astroid.parse(code)
        tree.lines = code.splitlines()
        lineno = 8

        output = line_number_to_line_index(tree, lineno)

        self.assertEqual(output, 7)

        insert_lines(tree, 0, ["import none", "import done"])

        output = line_number_to_line_index(tree, lineno)

        self.assertEqual(output, 9)

    def test_check_if_class(self):
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

        tree = astroid.parse(code)

        self.assertEqual(check_if_class(tree.body[0]), False)
        self.assertEqual(check_if_class(tree.body[1]), False)
        self.assertEqual(check_if_class(tree.body[2]), False)
        self.assertEqual(check_if_class(tree.body[3]), True)
        self.assertEqual(check_if_class(tree.body[4]), False)

    def test_check_if_function(self):
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

        tree = astroid.parse(code)

        self.assertEqual(check_if_function(tree.body[0]), True)
        self.assertEqual(check_if_function(tree.body[1]), True)
        self.assertEqual(check_if_function(tree.body[2]), True)
        self.assertEqual(check_if_function(tree.body[3]), False)
        self.assertEqual(check_if_function(tree.body[4]), False)

    def test_check_if_return(self):
        self.assertEqual(
            check_if_return(astroid.Return(0, 0, None, end_lineno=0, end_col_offset=0)),
            True,
        )
        self.assertEqual(
            check_if_return(astroid.If(0, 0, None, end_lineno=0, end_col_offset=0)),
            False,
        )

    def test_check_if_function_returns(self):
        code = """
            def f():
                pass
            
            def x():
                return

            def y():
                return 1
            
            def z():
                return None

            def a(x):
                if x == 0:
                    return True
                return None

            def b(x):
                def c():
                    if x == 0:
                        return True
                c()
                
            def c(x):
                if x == 0:
                    return True
                else:
                    return False
                
        """

        tree = astroid.parse(code)

        self.assertEqual(check_if_function_returns(tree.body[0]), False)
        self.assertEqual(check_if_function_returns(tree.body[1]), False)
        self.assertEqual(check_if_function_returns(tree.body[2]), True)
        self.assertEqual(check_if_function_returns(tree.body[3]), True)
        self.assertEqual(check_if_function_returns(tree.body[4]), True)
        self.assertEqual(check_if_function_returns(tree.body[5]), False)
        self.assertEqual(check_if_function_returns(tree.body[6]), True)

    def test_create_check_if_import(self):
        name = "none"

        output = create_check_if_import(name)

        self.assertEqual(output(astroid.Import([("none", None)])), True)
        self.assertEqual(output(astroid.Import([("torch", None)])), False)
        self.assertEqual(
            output(astroid.If(0, 0, None, end_lineno=0, end_col_offset=0)), False
        )

    def test_combine_checks(self):
        checks = [check_if_function, check_if_function_returns]

        output = combine_checks(checks)

        code = """
            def f():
                pass
            
            def x():
                return

            def y():
                return 1
            
            class z():
                pass
        """

        tree = astroid.parse(code)

        self.assertEqual(output(tree.body[0]), False)
        self.assertEqual(output(tree.body[1]), False)
        self.assertEqual(output(tree.body[2]), True)
        self.assertEqual(output(tree.body[3]), False)

    def test_write_tree(self):
        code = """
            def f():
                pass
            
            def x():
                return

            def y():
                return 1
            
            class z():
                pass
        """

        tree = astroid.parse(code)
        tree.lines = code.splitlines()

        dir_path = "./tests/temp"
        path = f"{dir_path}/write_tree_test.py"

        os.makedirs(dir_path, exist_ok=True)

        write_tree(tree, path)

        with open(path, "r") as f:
            tree2 = astroid.parse(f.read())

        shutil.rmtree(dir_path)

        self.assertEqual(len(tree.body), len(tree2.body))


class TestFakeInitFunction(unittest.TestCase):

    def test_constructor(self):
        fake_init_function = FakeInitFunction()

        self.assertEqual(fake_init_function.name, "__init__")
        self.assertEqual(len(fake_init_function.args.args), 1)
