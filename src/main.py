import os

from core import get_args, select_option
from exceptions import NoSelectionException
from py import fire, unittest
from fire import Fire

COMMANDS = {
    "add": ["fire", "unittest"]
}

def select_comand(group: str):
    options = COMMANDS[group]
    index = select_option(options, "Select command")

    if index is None:
        return NoSelectionException("")

    return options[index]


def add(command: str = "", *args, project=".", **kwargs):

    if command == "":
        command = select_comand("add")

    if command == "fire":
        command_args = get_args(args, kwargs, fire.EXPECTED_ARGS["add fire"], fire.DEFAULT_VALUES["add fire"])
        command_args = fire.verify_and_fix_args(command_args, project=project)
        fire.add_fire_to_file(**command_args)
    elif command == "unittest":
        command_args = get_args(args, kwargs, unittest.EXPECTED_ARGS["add unittest"], unittest.DEFAULT_VALUES["add unittest"])
        command_args = unittest.verify_and_fix_args(command_args, project=project)
        unittest.add_unittests_to_folder(**command_args, project=project)
    else:
        print(f"Unknown command {command}")
        return add("", *args, project=project, **kwargs)

    print()


if __name__ == "__main__":
    Fire()
