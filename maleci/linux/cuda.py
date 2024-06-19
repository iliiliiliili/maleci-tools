from typing import List
import astroid
import os
from pathlib import Path

from maleci.exceptions import CancelException, NotFolderException, WrongVersionException
from maleci.core import (
    resolve_path,
    write_lines,
    select_continue,
    select_option,
)

EXPECTED_ARGS = {
    "linux install cuda": [
        ("versions", "v", "version", "ver"),
        ("architecture", "arch"),
        ("distribution", "dist"),
        ("distribution_version", "dist_v", "dv"),
        ("installer_type", "installer"),
        ("toolkit_only", "toolkit"),
        ("temp", "tmp"),
        # ("delete_install_files", "del"),
        ("cuda_path", "cuda"),
        ("sudo"),
    ],
}

DEFAULT_VALUES = {
    "linux install cuda": {
        "versions": ["12.5", "12.1", "11.6"],
        "architecture": "x86_64",
        "distribution": "Ubuntu",
        "distribution_version": "20.04",
        "installer_type": "runfile",
        "temp": "~/install/maleci-cuda",
        "toolkit_only": True,
        # "delete_install_files": False,
        "cuda_path": "/usr/local",
        "sudo": "ask",
    },
}

INSTALL_PARAMS = {
    "12.5": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_12.5.0_555.42.02_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "12.4": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_12.4.1_550.54.15_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "12.3": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_12.3.2_545.23.08_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "12.2": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_12.2.2_535.104.05_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "12.1": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_12.1.1_530.30.02_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "12.0": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_12.0.1_525.85.12_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.8": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.8.0_520.61.05_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.7": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.7.1_515.65.01_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.6": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda_11.6.1_510.47.03_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.6.1_510.47.03_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.5": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.5.2_495.29.05_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.4": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.4.4_470.82.01_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.3": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.3.1_465.19.01_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.2": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.2.1_460.32.03_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.1": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.1.0_455.23.05_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "11.0": {
        "x86_64": {
            "Ubuntu": {
                "20.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_11.0.3_450.51.06_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "10.2": {
        "x86_64": {
            "Ubuntu": {
                "18.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_10.2.89_440.33.01_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
    "10.1": {
        "x86_64": {
            "Ubuntu": {
                "18.04": {
                    "runfile": {
                        "script_url": "https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run",
                        "commands": lambda toolkit_only: [f"sh cuda_10.1.243_418.87.00_linux.run --silent {'--tolkit' if toolkit_only else ''}"],
                    }
                }
            }
        }
    },
}

CUDA_PREFIX = "cuda-"


def verify_and_fix_args_install(args):

    if not (isinstance(args["versions"], list) or isinstance(args["versions"], tuple)):
        args["versions"] = [args["versions"]]
    
    args["versions"] = [str(a) for a in args["versions"]]

    for version in args["versions"]:

        params = INSTALL_PARAMS

        if version not in params:
            raise WrongVersionException(version, params.keys())
        params = params[version]

        if args["architecture"] not in params:
            raise WrongVersionException(
                args["architecture"], params.keys(), "architecture"
            )
        params = params[args["architecture"]]

        if args["distribution"] not in params:
            raise WrongVersionException(
                args["distribution"], params.keys(), "distribution"
            )
        params = params[args["distribution"]]

        if args["distribution_version"] not in params:
            raise WrongVersionException(
                args["distribution_version"], params.keys(), "distribution_version"
            )
        params = params[args["distribution_version"]]

        if args["installer_type"] not in params:
            raise WrongVersionException(
                args["installer_type"], params.keys(), "installer_type"
            )
        params = params[args["installer_type"]]

    return args


def find_cuda_versions(cuda_path):
    cuda_dirs = [a for a in os.listdir(cuda_path) if CUDA_PREFIX in a]
    cuda_versions = [a.replace(CUDA_PREFIX, "") for a in cuda_dirs]

    return cuda_versions


def install_cuda(
    versions,
    architecture,
    distribution,
    distribution_version,
    installer_type,
    toolkit_only,
    temp,
    # delete_install_files,
    cuda_path,
    sudo,
):

    temp_path = resolve_path(temp)

    if not select_continue(
        f"Selected cuda versions to be installed: {', '.join(versions)}\n" +
        f"architecture = {architecture}, distribution = {distribution},\n" + 
        f"distribution_version = {distribution_version}, installer_type = {installer_type}"
    ):
        raise CancelException()

    action_script_path = temp_path / "install-cuda.sh"

    if sudo == "ask":
        sudo = (
            select_option(
                [
                    f"Use sudo to install cuda",
                    f"Do not use sudo, create a script ({action_script_path}) to be used by a sudo-user to install cuda",
                ],
                "Use sudo?",
            )
            == 0
        )
    else:
        sudo = isinstance(sudo, bool) and sudo == True

    os.makedirs(temp_path, exist_ok=True)

    existing_versions = find_cuda_versions(cuda_path)

    action_script_lines = [
        "#!/bin/bash",
        "",
        f"cd {temp_path}",
        "",
    ]

    for version in versions:

        if version in existing_versions:
            print(f"Cuda version {version} already exists in {cuda_path}")
            continue

        params = INSTALL_PARAMS[version][architecture][distribution][
            distribution_version
        ][installer_type]

        if sudo:
            os.chdir(temp_path)
            os.system(f"wget {params['script_url']} --no-clobber")

            for command in params["commands"](toolkit_only):
                os.system(f"sudo {command}")

            print(f"Installed {CUDA_PREFIX}{version}")
        else:
            action_script_lines.append(f"# {CUDA_PREFIX}{version}")
            action_script_lines.append(f"wget {params['script_url']} --no-clobber")

            for command in params["commands"](toolkit_only):
                action_script_lines.append(command)

            action_script_lines.append("")

    if not sudo:
        write_lines(action_script_lines, action_script_path)
        print(
            f"Script is created at {action_script_path}. Run it with sudo to perform the installation"
        )
