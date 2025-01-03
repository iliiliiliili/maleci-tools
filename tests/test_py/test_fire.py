import os
from pathlib import Path
import shutil
import unittest

from maleci.core import get_args, find_code_files_in_folder
from maleci.py.core import (
    parse_file,
    find_nodes,
    create_check_if_import,
)
from maleci.py.fire import (
    verify_and_fix_args,
    check_if_fire_call,
    check_if_name_main_call,
    add_fire_to_file,
    EXPECTED_ARGS,
    DEFAULT_VALUES,
)
class TestFire(unittest.TestCase):

    def test_verify_and_fix_args(self):
        args = get_args([], {"file":"single.py"}, EXPECTED_ARGS["py add fire"], DEFAULT_VALUES["py add fire"])
        project = "./tests/test_py"

        output = verify_and_fix_args(args, project)

        self.assertEqual(output, {
            "path": str(Path("./tests/test_py/single.py").absolute()),
            "silent": False,
            "requirements_path": str(Path("./tests/test_py/requirements.txt").absolute()),
        })


    def test_add_fire_to_file(self):

        files = [a[1] for a in find_code_files_in_folder("./tests/py/inputs/fire")]

        for file in files:
            
            dir_path = "./tests/temp"
            path = f"{dir_path}/write_file.py"

            os.makedirs(dir_path, exist_ok=True)

            shutil.copyfile(file, path)
            add_fire_to_file(path, silent=True, backup=False)

            tree = parse_file(path, module_name="tmp")

            self.assertEqual(len(find_nodes(tree, create_check_if_import("fire"))), 1)

            name_main_nodes = find_nodes(tree, check_if_name_main_call)

            self.assertEqual(len(name_main_nodes), 1)
            self.assertEqual(len(find_nodes(name_main_nodes[0], check_if_fire_call)), 1)

            shutil.rmtree(dir_path)
