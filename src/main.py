import os

from core import get_args, select_option
from py import fire
from fire import Fire

COMMANDS = {
    "add": ["fire", "unittest"]
}

def select_comand(group: str):
    options = COMMANDS[group]
    index = select_option(options, "Select command")

    if index is None:
        return ValueError("")

    return options[index]


def add(command: str = "", *args, project=".", **kwargs):

    if command == "":
        command = select_comand("add")

    if command == "fire":
        command_args = get_args(args, kwargs, fire.EXPECTED_ARGS["add fire"], fire.DEFAULT_VALUES["add fire"])
        command_args = fire.verify_and_fix_args(command_args, project=project)
        fire.add_fire_to_file(**command_args)

    print()


if __name__ == "__main__":
    Fire()
