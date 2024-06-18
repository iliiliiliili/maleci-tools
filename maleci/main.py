import os

from maleci.core import get_args, select_option
from maleci.exceptions import NoSelectionException
from maleci.py import fire, unittest
from maleci.linux import lmod
from fire import Fire

COMMANDS = {
    "add": ["fire", "unittest"],
    "linux": ["add", "init", "install"],
    "linux add": ["lmod", ],
    "linux add lmod": ["cuda", ],
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
        command_args = get_args(
            args,
            kwargs,
            fire.EXPECTED_ARGS["add fire"],
            fire.DEFAULT_VALUES["add fire"],
        )
        command_args = fire.verify_and_fix_args(command_args, project=project)
        fire.add_fire_to_file(**command_args)
    elif command == "unittest":
        command_args = get_args(
            args,
            kwargs,
            unittest.EXPECTED_ARGS["add unittest"],
            unittest.DEFAULT_VALUES["add unittest"],
        )
        command_args = unittest.verify_and_fix_args(command_args, project=project)
        unittest.add_unittests_to_folder(**command_args, project=project)
    else:
        print(f"Unknown command {command}")
        return add("", *args, project=project, **kwargs)

    print()


def linux_add_lmod(command: str = "", *args, **kwargs):
    if command == "":
        command = select_comand("linux add lmod")
    
    if command == "cuda":
        command_args = get_args(
            args,
            kwargs,
            lmod.EXPECTED_ARGS["linux add lmod cuda"],
            lmod.DEFAULT_VALUES["linux add lmod cuda"],
        )
        command_args = lmod.verify_and_fix_args_add(command_args)
        lmod.add_cuda_modulefiles(**command_args)


def linux_add(command: str = "", *args, **kwargs):
    if command == "":
        command = select_comand("linux add")
    
    if command == "lmod":
        linux_add_lmod(*args, **kwargs)
    else:
        print(f"Unknown command {command}")
        return linux_add("", *args, **kwargs)


def linux(command: str = "", *args, **kwargs):

    if command == "":
        command = select_comand("linux")
    
    if command == "add":
        linux_add(*args, **kwargs)
    else:
        print(f"Unknown command {command}")
        return linux("", *args, **kwargs)



if __name__ == "__main__":
    Fire()
