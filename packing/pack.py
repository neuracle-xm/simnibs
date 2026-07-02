"""
SimNIBS 打包入口模块

打包命令
-----------
python packing/pack.py --env simnibs_env

说明：
- 会先临时克隆已有 conda 环境
- 会先清空 `dist/`，再自动重新构建当前版本对应的 wheel
- 再将临时环境中的 editable `simnibs` 替换为 wheel 安装
- 打包完成后自动删除临时克隆环境
"""

import argparse
import glob
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools_scm import ScmVersion, get_version


def _env_python(env_prefix: str) -> str:
    candidates = []

    if sys.platform == "win32":
        candidates = [
            Path(env_prefix) / "python.exe",
            Path(env_prefix) / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            Path(env_prefix) / "bin" / "python",
        ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find python executable in cloned env. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def _remove_pip_packages(env_prefix: str) -> None:
    python_exe = _env_python(env_prefix)
    pip_remove = [
        "pyqt5",
        "pyqt5-qt5",
        "pyqt5-sip",
        "cython",
    ]
    print("Using python:", python_exe)
    subprocess.run(
        [python_exe, "-m", "pip", "list"],
        check=False,
    )
    subprocess.run(
        [python_exe, "-m", "pip", "uninstall", "-y", *pip_remove],
        check=False,
    )


def _remove_pyqt5_cython_after_unpack(packed_env_dir: str) -> None:
    """解包后兜底删除"""
    site = Path(packed_env_dir) / "Lib" / "site-packages"
    patterns = [
        "PyQt5",
        "PyQt5-*",
        "pyqt5*",
        "Cython",
        "Cython-*",
        "cython.py",
    ]
    for pattern in patterns:
        for path in site.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                print(f"Removed dir: {path}")
            elif path.exists():
                path.unlink()
                print(f"Removed file: {path}")


def _shell_quote(value: str) -> str:
    return subprocess.list2cmdline([value])


def _build_simnibs_wheel(simnibs_root_dir: str, version: str) -> str:
    """
    清理 dist 目录并重新构建当前版本的 wheel。

    Parameters
    ----------
    simnibs_root_dir : str
        SimNIBS 仓库根目录
    version : str
        当前目标版本号

    Returns
    -------
    str
        构建产出的 wheel 绝对路径
    """
    wheel_dir = Path(simnibs_root_dir) / "dist"
    if wheel_dir.exists():
        shutil.rmtree(wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building SimNIBS wheel for version {version}")
    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(wheel_dir)],
        check=True,
        cwd=simnibs_root_dir,
    )
    wheels = sorted(wheel_dir.glob(f"simnibs-{version}-*.whl"))
    assert len(wheels) == 1, (
        f"Expected exactly one SimNIBS wheel for version {version} in {wheel_dir}, "
        f"found {len(wheels)}"
    )
    return str(wheels[0])


def _resolve_simnibs_wheel(
    simnibs_root_dir: str,
    version: str,
    simnibs_wheel: str | None,
    required: bool,
) -> str | None:
    if simnibs_wheel is not None:
        wheel_path = Path(simnibs_wheel)
        assert wheel_path.exists(), f"SimNIBS wheel does not exist: {wheel_path}"
        return str(wheel_path)
    auto_wheel = _build_simnibs_wheel(simnibs_root_dir, version)
    if required:
        assert auto_wheel is not None, (
            "A SimNIBS wheel is required but wheel build did not produce a usable artifact"
        )
    return auto_wheel


def _create_temp_env_from_file(
    env_name: str, env_prefix: str, simnibs_wheel: str
) -> None:
    print(f"Creating temporary environment from {env_name}")
    subprocess.run(
        f"conda env create -p {_shell_quote(env_prefix)} -f {_shell_quote(env_name)} -y",
        check=True,
        shell=True,
    )
    print("Installing SimNIBS")
    subprocess.run(
        f"conda run -p {_shell_quote(env_prefix)} python -m pip install --no-deps {_shell_quote(simnibs_wheel)}",
        check=True,
        shell=True,
    )


def _clone_existing_env(source_env_name: str, env_prefix: str) -> None:
    print(f"Cloning existing environment: {source_env_name}")
    subprocess.run(
        f"conda create -y -p {_shell_quote(env_prefix)} --clone {_shell_quote(source_env_name)}",
        check=True,
        shell=True,
    )


def _replace_editable_simnibs_with_wheel(env_prefix: str, simnibs_wheel: str) -> None:
    print("Replacing editable SimNIBS with wheel")
    subprocess.run(
        f"conda run -p {_shell_quote(env_prefix)} python -m pip uninstall -y simnibs",
        check=True,
        shell=True,
    )
    subprocess.run(
        f"conda run -p {_shell_quote(env_prefix)} python -m pip install --no-deps {_shell_quote(simnibs_wheel)}",
        check=True,
        shell=True,
    )


def _remove_conda_env(env_prefix: str) -> None:
    if not os.path.isdir(env_prefix):
        return
    print("Removing temporary environment")
    subprocess.run(
        f"conda env remove -y -p {_shell_quote(env_prefix)}",
        check=True,
        shell=True,
    )


def _increment_version_dev_branch(
    version: ScmVersion,
    major_increment: int = 0,
    minor_increment: int = 1,
    patch_increment: int = 0,
) -> str:
    # Get version parts
    increments = [major_increment, minor_increment, patch_increment]
    parts_orig = [i for i in str(version.tag).split(".")]
    if len(parts_orig) > 3 or len(parts_orig) < 1:
        raise ValueError(f"{version} is not in the correct format X.Y.Z")

    parts_new = [0] * len(parts_orig)
    for i, p in enumerate(parts_orig):
        try:
            temp_num = int(p)
            parts_new[i] = temp_num + increments[i]
        except:
            # find digits
            m = re.search(r"\d+", p)
            # No digits (should not happen)
            if m is None:
                continue
            temp_num = int(p[m.start() : m.end()])
            parts_new[i] = temp_num + increments[i]

    if all(v == 0 for v in parts_new):
        print("Could not update version number")
        new_version = str(version.tag)
    else:
        new_version = ".".join(str(i) for i in parts_new)

    return new_version


def custom_version_func(version: ScmVersion) -> str:
    return version.format_with("{tag}")


def _ignore_neuracle_copy(src: str, names: list[str]) -> set[str]:
    """
    过滤不需要复制的 neuracle 文件。

    Parameters
    ----------
    src : str
        当前复制源目录
    names : list[str]
        当前目录下的名称列表

    Returns
    -------
    set[str]
        需要忽略的名称集合
    """
    ignored = {"__pycache__", "private_gitignore", "docs", "demo", "validation"}
    if Path(src).name == "neuracle":
        ignored.update({"CLAUDE.md"})
    if Path(src).name == "atlas":
        ignored.update(
            {
                "atlas",
                "build_standardized_registry.py",
                "generate_standardized_rois.py",
                "standardize_atlases.py",
                "validate_standardized_atlases.py",
                "structure.md",
            }
        )
    return {name for name in names if name in ignored}


def _copy_neuracle_runtime_to_root(simnibs_root_dir: str, pack_dir: str) -> None:
    """
    将 neuracle 运行期目录复制到打包根目录。

    Parameters
    ----------
    simnibs_root_dir : str
        仓库根目录
    pack_dir : str
        最终打包目录
    """
    source_neuracle_dir = Path(simnibs_root_dir) / "neuracle"
    target_neuracle_dir = Path(pack_dir) / "neuracle"
    shutil.copytree(
        source_neuracle_dir,
        target_neuracle_dir,
        ignore=_ignore_neuracle_copy,
    )


def _copy_license_to_root(simnibs_root_dir: str, pack_dir: str) -> None:
    """
    将仓库根目录的 LICENSE.txt 复制到打包根目录。

    Parameters
    ----------
    simnibs_root_dir : str
        仓库根目录
    pack_dir : str
        最终打包目录
    """
    source_license = Path(simnibs_root_dir) / "LICENSE.txt"
    target_license = Path(pack_dir) / "LICENSE.txt"
    shutil.copy(source_license, target_license)


def _write_commit_marker_to_root(simnibs_root_dir: str, pack_dir: str) -> None:
    """
    在打包根目录生成以当前 Git 提交号命名的标记文件。

    Parameters
    ----------
    simnibs_root_dir : str
        仓库根目录
    pack_dir : str
        最终打包目录
    """
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=simnibs_root_dir,
        text=True,
    ).strip()
    target_marker = Path(pack_dir) / f"{commit_hash}.txt"
    target_marker.write_text(f"{commit_hash}\n", encoding="utf-8")


def build(
    env_name: str | None = None,
    simnibs_wheel: str | None = None,
    # pack_dir: str | None = None,
    macos_developer_id=None,
):
    # Import these here so I don't need to install them when running setup.py
    # The reason is that I need the two functions above in setup.py and here
    # but I don't want to duplicate them. So setup.py imports them from here.
    import conda_pack
    from jinja2 import Template

    simnibs_root_dir = os.path.normpath(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "..")
    )
    version = get_version(
        git_describe_command="git describe --tags --abbrev=0",
        version_scheme=custom_version_func,
    )

    pack_dir = os.path.abspath("simnibs_package")
    if os.path.isdir(pack_dir):
        shutil.rmtree(pack_dir)

    env_is_temporary = False
    env_prefix = None
    simnibs_wheel = _resolve_simnibs_wheel(
        simnibs_root_dir,
        version,
        simnibs_wheel,
        required=True,
    )

    try:
        if Path(env_name).is_file():
            assert simnibs_wheel is not None
            env_is_temporary = True
            env_prefix = os.path.join(pack_dir, "simnibs_env_tmp")
            _create_temp_env_from_file(env_name, env_prefix, simnibs_wheel)
            # when env_name refers to a file, we create a temp. environment and
            # specify the env prefix
            env_name = None
        else:
            print(f"Using existing environment: {env_name}")
            res = subprocess.run(
                f"conda list -n {_shell_quote(env_name)} simnibs --json",
                shell=True,
                capture_output=True,
            )
            assert len(json.loads(res.stdout)) == 1, (
                f"SimNIBS is not installed in {env_name}"
            )
            assert simnibs_wheel is not None
            env_is_temporary = True
            env_prefix = os.path.join(pack_dir, "simnibs_env_clone")
            _clone_existing_env(env_name, env_prefix)
            _replace_editable_simnibs_with_wheel(env_prefix, simnibs_wheel)
            _remove_pip_packages(env_prefix)
            _remove_unneeded_conda_packages(env_prefix)
            env_name = None

        print("Packing environment...")
        # (use .tar because MacOS erases the execute permission in .zip)
        packed_env_file = os.path.join(pack_dir, "simnibs_env.tar")
        packed_env_dir = os.path.join(pack_dir, "simnibs_env")
        conda_pack.pack(
            name=env_name,
            prefix=env_prefix,
            output=packed_env_file,
            compress_level=0,
            force=True,
            verbose=False,
            ignore_missing_files=True,
        )

        print("Unpacking environment")
        shutil.unpack_archive(
            packed_env_file,
            packed_env_dir,
        )
        os.remove(packed_env_file)
        _remove_pyqt5_cython_after_unpack(packed_env_dir)
        _remove_pdb_files(pack_dir)
        print("Patching unpacked environment")
        shutil.copy(
            os.path.join(simnibs_root_dir, "packing", "fix_entrypoints.py"),
            packed_env_dir,
        )
        _copy_neuracle_runtime_to_root(simnibs_root_dir, pack_dir)
        _copy_license_to_root(simnibs_root_dir, pack_dir)
        _write_commit_marker_to_root(simnibs_root_dir, pack_dir)

        # Create OS-specific installer
        if sys.platform == "win32":
            # Move the sitecustomize.py file to the site-packages directory
            # This should allow for using the python interpreter without activating the environment
            shutil.copy(
                os.path.join(
                    simnibs_root_dir,
                    "simnibs",
                    "_internal_resources",
                    "sitecustomize.py",
                ),
                os.path.join(pack_dir, "simnibs_env", "Lib", "site-packages"),
            )
            print(f"Windows 打包完成，输出目录: {pack_dir}")
        elif sys.platform == "darwin":
            installer_name = "simnibs_installer_macos.pkg"
            with tempfile.TemporaryDirectory() as tmpdir:
                for fn in glob.glob(
                    os.path.join(simnibs_root_dir, "packing", "macOS_installer", "*")
                ):
                    fn_out = os.path.join(tmpdir, os.path.basename(fn))
                    with open(fn, "r") as f:
                        template = Template(f.read()).render(
                            version=".".join(version.split(".")[:2]),
                            full_version=version,
                        )
                    with open(fn_out, "w") as f:
                        f.write(template)
                    os.chmod(fn_out, os.stat(fn).st_mode)

                # Workaroud for Notarization
                # Instead of signing all binaries, I zip the enironment with a password
                # The postinstall script will unzip it in the user's computer
                print("Repacking...")
                orig_folder = os.path.abspath(os.curdir)
                os.chdir(pack_dir)
                subprocess.run(
                    [
                        "zip",
                        "-y",
                        "-q",
                        "-P",
                        "password",
                        "-r",
                        "simnibs_env.zip",
                        "simnibs_env",
                    ]
                )
                os.chdir(orig_folder)
                shutil.rmtree(os.path.join(pack_dir, "simnibs_env"))

                print("Running pkgbuild")
                subprocess.run(
                    [
                        "pkgbuild",
                        "--root",
                        pack_dir,
                        "--identifier",
                        f"org.SimNIBS.{version}",
                        "--version",
                        version,
                        "--scripts",
                        tmpdir,
                        "--install-location",
                        "/Applications/SimNIBS-" + ".".join(version.split(".")[:2]),
                        os.path.join(tmpdir, installer_name),
                    ],
                    check=True,
                )
                print("Running productbuild")
                if macos_developer_id is not None:
                    sign = ["--sign", macos_developer_id]
                else:
                    sign = []
                subprocess.run(
                    [
                        "productbuild",
                        "--distribution",
                        os.path.join(tmpdir, "Distribution"),
                        "--package-path",
                        tmpdir,
                        "--resources",
                        tmpdir,
                        installer_name,
                    ]
                    + sign,
                    check=True,
                )
        elif sys.platform == "linux":
            # Write the install script
            fn_script = os.path.join(pack_dir, "install")
            with open(os.path.join(simnibs_root_dir, "packing", "install"), "r") as f:
                install_script = Template(f.read()).render(
                    version=".".join(version.split(".")[:2]), full_version=version
                )
            with open(fn_script, "w") as f:
                f.write(install_script)
            os.chmod(
                fn_script,
                os.stat(fn_script).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
            print("Repacking...")
            installer_name = "simnibs_installer_linux"
            shutil.make_archive(
                installer_name,
                "gztar",
                # I use root_dir and base_dir so that it decompresses into a folder called
                # simnibs_installer
                root_dir=".",
                base_dir=os.path.relpath(pack_dir),
            )
    finally:
        if env_is_temporary and env_prefix is not None:
            _remove_conda_env(env_prefix)

    # Remove the (temporary) packing dir
    if sys.platform != "win32":
        shutil.rmtree(pack_dir)

    # print(f"Created installer {installer_file}")


def _remove_unneeded_conda_packages(env_prefix: str) -> None:
    """Remove conda packages that are not needed in the final package."""
    conda_remove = [
        "conda-pack",
        "freeglut",
        "jupyterlab",
        "libwebp",
        "mock",
        "pyopengl",
        "pytest",
        "tbb-devel",
    ]

    if conda_remove:
        print(f"删除冗余的conda包:{conda_remove}")
        subprocess.run(
            f"conda remove -y -p {_shell_quote(env_prefix)} "
            + " ".join(_shell_quote(p) for p in conda_remove),
            check=True,
            shell=True,
        )


def _remove_pdb_files(root_dir: str) -> None:
    """
    删除打包目录中的所有 .pdb 调试符号文件。

    Parameters
    ----------
    root_dir : str
        需要扫描的根目录
    """
    root = Path(root_dir)

    removed_count = 0
    removed_size = 0

    for path in root.rglob("*.pdb"):
        try:
            size = path.stat().st_size

            path.unlink()

            removed_count += 1
            removed_size += size

            print(f"Removed: {path}")

        except OSError as e:
            print(f"Failed to remove {path}: {e}")

    print(
        f"Removed {removed_count} .pdb files, freed {removed_size / 1024 / 1024:.1f} MB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="simnibs-pack", description="Create SimNIBS installers."
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Name of existing environment to use or the name of a conda environment file (yaml). In the latter case, a temporary environment will be created.",
    )
    parser.add_argument(
        "--simnibs-wheel",
        default=None,
        help="Directory with the SimNIBS wheels to be packed",
    )
    parser.add_argument(
        "--macos-developer-id",
        default=None,
        help="Developer ID for signing in MacOS, DOES NOT SUPPORT NOTARIZATION (optional)",
    )
    args = parser.parse_args(sys.argv[1:])
    build(args.env, args.simnibs_wheel, args.macos_developer_id)
