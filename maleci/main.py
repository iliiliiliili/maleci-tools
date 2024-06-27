import os

from maleci.core import get_args, select_option
from maleci.exceptions import NoSelectionException
from maleci.py import fire, pip, unittest, torch
from maleci.linux import lmod, cuda
from fire import Fire

COMMANDS = {
    "": ["add", "init", "pip", "linux", "py"],
    "py": ["add", "init", "pip"],
    "py add": ["fire", "unittest"],
    "py init": ["torch"],
    "py init torch": ["empty", "mnist"],
    "py pip": ["install"],
    "linux": ["add", "init", "install"],
    "linux add": [
        "lmod",
    ],
    "linux add lmod": [
        "cuda",
    ],
    "linux install": ["lmod", "cuda"],
}


def select_comand(group: str):
    options = COMMANDS[group]
    index = select_option(options, "Select command")

    if index is None:
        return NoSelectionException("")

    return options[index]


def py_add(command: str = "", *args, project=".", **kwargs):

    if command == "":
        command = select_comand("py add")

    if command == "fire":
        command_args = get_args(
            args,
            kwargs,
            fire.EXPECTED_ARGS["py add fire"],
            fire.DEFAULT_VALUES["py add fire"],
        )
        command_args = fire.verify_and_fix_args(command_args, project=project)
        fire.add_fire_to_file(**command_args)
    elif command == "unittest":
        command_args = get_args(
            args,
            kwargs,
            unittest.EXPECTED_ARGS["py add unittest"],
            unittest.DEFAULT_VALUES["py add unittest"],
        )
        command_args = unittest.verify_and_fix_args(command_args, project=project)
        unittest.add_unittests_to_folder(**command_args, project=project)
    else:
        print(f"Unknown command {command}")
        return py_add("", *args, project=project, **kwargs)

    print()


def py_init_torch(command: str = "", *args, project=".", **kwargs):

    if command == "":
        command = select_comand("py init torch")

    if command in ["empty", "pytorch", "."]:
        command_args = get_args(
            args,
            kwargs,
            torch.EXPECTED_ARGS["py init torch empty"],
            torch.DEFAULT_VALUES["py init torch empty"],
        )
        command_args = torch.verify_and_fix_args_init_empty(command_args, project=project)
        torch.init_pytorch_empty(**command_args, project=project)
    elif command in ["mnist"]:
        command_args = get_args(
            args,
            kwargs,
            torch.EXPECTED_ARGS["py init torch mnist"],
            torch.DEFAULT_VALUES["py init torch mnist"],
        )
        command_args = torch.verify_and_fix_args_init_mnist(command_args, project=project)
        torch.init_pytorch_mnist(**command_args, project=project)
    else:
        print(f"Unknown command {command}")
        return py_init("", *args, project=project, **kwargs)

    print()

def py_init(command: str = "", *args, project=".", **kwargs):

    if command == "":
        command = select_comand("py init")

    if command in ["torch", "pytorch"]:
        py_init_torch(*args, project=".", **kwargs)
    else:
        print(f"Unknown command {command}")
        return py_init("", *args, project=project, **kwargs)

    print()


def py_pip(command: str = "", *args, project=".", **kwargs):

    if command == "":
        command = select_comand("py pip")

    if command in ["install", "i"]:
        command_args = get_args(
            args,
            kwargs,
            pip.EXPECTED_ARGS["py pip install"],
            pip.DEFAULT_VALUES["py pip install"],
        )
        command_args = pip.verify_and_fix_args(command_args, project=project)
        pip.pip_install(**command_args)
    else:
        print(f"Unknown command {command}")
        return py_pip("", *args, project=project, **kwargs)

    print()


def py(command: str = "", *args, project=".", **kwargs):

    if command == "":
        command = select_comand("py")

    if command == "add":
        py_add(*args, project=project, **kwargs)
    elif command == "init":
        py_init(*args, project=project, **kwargs)
    else:
        print(f"Unknown command {command}")
        return py("", *args, **kwargs)


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

    if command in ["lmod", "modules"]:
        linux_add_lmod(*args, **kwargs)
    else:
        print(f"Unknown command {command}")
        return linux_add("", *args, **kwargs)


def linux_install(command: str = "", *args, **kwargs):
    if command == "":
        command = select_comand("linux install")

    if command in ["lmod", "modules"]:
        command_args = get_args(
            args,
            kwargs,
            lmod.EXPECTED_ARGS["linux install lmod"],
            lmod.DEFAULT_VALUES["linux install lmod"],
        )
        command_args = lmod.verify_and_fix_args_install(command_args)
        lmod.install_lmod(**command_args)
    elif command == "cuda":
        command_args = get_args(
            args,
            kwargs,
            cuda.EXPECTED_ARGS["linux install cuda"],
            cuda.DEFAULT_VALUES["linux install cuda"],
        )
        command_args = cuda.verify_and_fix_args_install(command_args)
        cuda.install_cuda(**command_args)
    else:
        print(f"Unknown command {command}")
        return linux_install("", *args, **kwargs)


def linux(command: str = "", *args, **kwargs):

    if command == "":
        command = select_comand("linux")

    if command == "add":
        linux_add(*args, **kwargs)
    elif command == "install":
        linux_install(*args, **kwargs)
    else:
        print(f"Unknown command {command}")
        return linux("", *args, **kwargs)


def main(command: str = "", *args, **kwargs):

    if command == "":
        command = select_comand("")

    if command == "add":
        py_add(*args, **kwargs)
    elif command == "init":
        py_init(*args, **kwargs)
    elif command == "pip":
        py_pip(*args, **kwargs)
    elif command == "py":
        py(*args, **kwargs)
    elif command == "linux":
        linux(*args, **kwargs)
    else:
        print(f"Unknown command {command}")
        return main("", *args, **kwargs)


if __name__ == "__main__":
    Fire(main)
