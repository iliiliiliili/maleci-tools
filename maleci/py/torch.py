import astroid
import os
from pathlib import Path

from maleci.core import (
    backup_file,
    find_files_in_folder,
    indent_single,
    select_option,
    resolve_path,
    path_in_project,
    get_relative_path,
    write_lines,
)

from maleci.py.core import DEFAULT_SPACES, add_to_requirements

EXPECTED_ARGS = {
    "py init torch": [
        ("version", "v"),
        ("name", "n"),
        ("python", "python_version", "pv"),
        ("cuda", "cuda_version", "cv"),
        ("use_lmod", "lmod", "use_modules", "modules", "use_module", "module"),
        ("install_script", "script", "install"),
        ("main_path", "main"),
        ("requirements_path", "requirements"),
        ("spaces"),
    ]
}

DEFAULT_VALUES = {
    "py init torch": {
        "version": None,
        "name": None,
        "python": "3.10",
        "cuda": "12.1",
        "use_lmod": True,
        "install_script": "install.sh",
        "main_path": "main.py",
        "requirements_path": "requirements.txt",
        "spaces": DEFAULT_SPACES,
    }
}

INSTALL_COMMANDS = {
    "latest": {
        "12.1": "conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia",
        "11.8": "conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia",
        "cpu": "conda install -y pytorch torchvision torchaudio cpuonly -c pytorch",
    },
    "2.3.0": {
        "12.1": "conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia",
        "11.8": "conda install -y pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia",
    },
    "2.2.2": {
        "12.1": "conda install -y pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia",
        "11.8": "conda install -y pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia",
    },
    "2.1.2": {
        "12.1": "conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia",
        "11.8": "conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia",
    },
    "2.0.1": {
        "12.1": "conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia",
        "11.7": "conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia",
    },
    "1.13.1": {
        "11.7": "conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia",
        "11.6": "conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia",
    },
    "1.12.1": {
        "11.6": "conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge",
        "11.3": "conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch",
        "10.2": "conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch",
    },
    "1.11.0": {
        "11.3": "conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch",
        "10.2": "conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch",
    },
    "1.10.1": {
        "11.3": "conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge",
        "10.2": "conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch",
    },
    "1.9.1": {
        "11.3": "conda install -y pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge",
        "10.2": "conda install -y pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch",
    },
}


SELECT_VERSION_MESSAGE = "Select pytorch version"
SELECT_CUDA_VERSION_MESSAGE = "Select cuda version"


def verify_and_fix_args_init(args, project):

    params = INSTALL_COMMANDS

    if args["version"] not in params:
        options = list(params.keys())
        index = select_option(options, SELECT_VERSION_MESSAGE)
        args["version"] = options[index]

    params = params[args["version"]]

    args["cuda"] = str(args["cuda"])

    if args["cuda"] not in params:
        options = list(params.keys())
        index = select_option(options, SELECT_CUDA_VERSION_MESSAGE)
        args["cuda"] = options[index]

    project_path = resolve_path(project)

    args["install_script"] = path_in_project(args["install_script"], project_path)
    args["main_path"] = path_in_project(args["main_path"], project_path)
    args["requirements_path"] = path_in_project(args["requirements_path"], project_path)

    if not os.path.isdir(project_path):
        raise ValueError("Project path is not a directory")

    if os.path.exists(args["install_script"]):
        raise ValueError(f"Install script already exists at {args['install_script']}")

    if os.path.exists(args["main_path"]):
        raise ValueError(f"Main file already exists at {args['main_path']}")

    return args


def create_py_code_for_init(main_path, requirements_path, spaces):

    main_lines = [
        "import torch",
        "from fire import Fire",
        "",
        "",
        "def main():",
        indent_single("print(torch.__version__)", spaces),
        indent_single("print(torch.version.cuda)", spaces),
        indent_single("device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')", spaces),
        indent_single("print(torch.rand([2, 6, 5], device=device))", spaces),
        "",
        "if __name__ == \"__main__\":",
        indent_single("Fire(main)", spaces),
        "",
    ]

    write_lines(main_lines, main_path)
    print(f"Main python script created at {main_path}.")
    add_to_requirements(["fire"], requirements_path)
    print(f"Requirements created at {requirements_path}.")
    


def init_pytorch(
    version,
    name,
    python,
    cuda,
    use_lmod,
    install_script,
    main_path,
    requirements_path,
    spaces,
    project,
):
    
    project_path = resolve_path(project)

    if name is None:
        name = Path(project).absolute().name

    install_lines = [
        "#!/bin/bash -i",
        "",
        f"conda create -n {name} python={python}",
        f"conda activate {name}",
        "",
        *([f"module load cuda/{cuda}", ""] if use_lmod else []),
        INSTALL_COMMANDS[version][cuda],
        "",
        f"pip install -r {get_relative_path(requirements_path, project_path)}",
    ]

    create_py_code_for_init(main_path, requirements_path, spaces)

    write_lines(install_lines, install_script)
    print(f"Install script created at {install_script}. To launch it, use:")
    print(f"cd {project}; bash -i {install_script}")
