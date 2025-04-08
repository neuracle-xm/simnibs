import stat
import glob
import json
import sys
import argparse
import os
import subprocess
import shutil
import tempfile
import re

from setuptools_scm import get_version
from setuptools_scm import ScmVersion

def _increment_version_dev_branch(
        version: ScmVersion, major_increment: int = 0, minor_increment: int = 1, patch_increment: int = 0
) -> str:

    # Get version parts
    increments = [major_increment, minor_increment, patch_increment]
    parts_orig = [i for i in str(version.tag).split(".")]
    if len(parts_orig) > 3 or len(parts_orig) < 1:
        raise ValueError(f"{version} is not in the correct format X.Y.Z")

    parts_new = [0]*len(parts_orig)
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
            temp_num = int(p[m.start():m.end()])
            parts_new[i] = temp_num + increments[i]

    if all(v==0 for v in parts_new):
        print('Could not update version number')
        new_version = str(version.tag)
    else:
        new_version = ".".join(str(i) for i in parts_new)

    return new_version

def custom_version_func(version: ScmVersion) -> str:
    if 'dev' in version.branch.lower():
        return version.format_next_version(_increment_version_dev_branch, "{guessed}")
    else:
        return version.format_with("{tag}")

def build(
    env_name: str | None = None,
    simnibs_wheel_dir: str | None = None,
    # pack_dir: str | None = None,
    macos_developer_id = None
):
    # Import these here so I don't need to install them when running setup.py
    # The reason is that I need the two functions above in setup.py and here
    # but I don't want to duplicate them. So setup.py imports them from here.
    import conda_pack
    from jinja2 import Template

    simnibs_root_dir = os.path.normpath(os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        '..'
    ))
    version = get_version(git_describe_command="git describe --tags --abbrev=0", version_scheme=custom_version_func)

    pack_dir = os.path.abspath('simnibs_installer')
    if os.path.isdir(pack_dir):
        shutil.rmtree(pack_dir)

    if env_name is None:
        assert simnibs_wheel_dir is not None, "You must specify a path to simnibs wheels when using a temporary environment"
        simnibs_wheel_dir = os.path.abspath(simnibs_wheel_dir)
        wheels = glob.glob(
            os.path.join(simnibs_wheel_dir, f'simnibs-{version}*.whl')
        )
        if len(wheels) == 0:
            raise FileNotFoundError(f'Did not find any wheels for simnibs {version} in {simnibs_wheel_dir}')

        # Create temporary environment
        env_is_temporary = True
        env_prefix = os.path.join(pack_dir, 'simnibs_env_tmp')

        if sys.platform == 'linux':
            os_name = 'linux'
        elif sys.platform == 'darwin':
            os_name = 'macOS'
        elif sys.platform == 'win32':
            os_name = 'win'
        else:
            raise OSError('OS not supported!')
        env_file = os.path.join(simnibs_root_dir, f'environment_{os_name}.yml')

        print(f"Creating temporary environment from {env_file}")
        subprocess.run(
            f'conda env create -p {env_prefix} -f {env_file} -y',
            check=True,
            shell=True
        )

        print("Installing SimNIBS")
        subprocess.run(
            f'conda run -p {env_prefix} python -m pip install simnibs=={version} --no-deps --no-index --find-links={simnibs_wheel_dir}',
            check=True,
            shell=True
        )

    else:
        # Use an existing environment
        env_is_temporary = False
        env_prefix = None
        print(f"Using existing environment: {env_name}")

        # check that simnibs is in the current environment
        res = subprocess.run("conda list simnibs --json", shell=True,  capture_output=True)
        assert len(json.loads(res.stdout)) == 1, f"SimNIBS is not installed in {env_name}"

    print("Packing environment...")
    # (use .tar because MacOS erases the execute permission in .zip)
    packed_env_file = os.path.join(pack_dir, 'simnibs_env.tar')
    packed_env_dir = os.path.join(pack_dir, 'simnibs_env')
    conda_pack.pack(
        name=env_name,
        prefix=env_prefix,
        output=packed_env_file,
        compress_level=0,
        force=True,
        verbose=True,
        ignore_missing_files=True
    )

    print("Unpacking environment")
    shutil.unpack_archive(packed_env_file, packed_env_dir,)
    os.remove(packed_env_file)

    print("Patching unpacked environment")
    shutil.copytree(
        os.path.join(simnibs_root_dir, 'docs', 'build', 'html'),
        os.path.join(pack_dir, 'documentation')
    )
    shutil.copy(
        os.path.join(simnibs_root_dir, 'packing', 'fix_entrypoints.py'),
        packed_env_dir
    )

    if env_is_temporary:
        print("Removing temporary environment")
        subprocess.run(
            f'conda env remove -y -p {env_prefix}',
            check=True,
            shell=True
        )

    # Create OS-specific installer
    if sys.platform == 'win32':
        # Move the sitecustomize.py file to the site-packages directory
        # This should allow for using the python interpreter without activating the environment
        shutil.copy(
            os.path.join(simnibs_root_dir, 'simnibs', '_internal_resources', 'sitecustomize.py'),
            os.path.join(pack_dir, 'simnibs_env', 'Lib', 'site-packages')
        )
        #Use the installer.nsi template to create an NSIS installer
        shutil.copy(
            os.path.join(simnibs_root_dir, 'simnibs', '_internal_resources',
                         'icons', 'simnibs', 'gui_icon.ico'),
            os.path.join(pack_dir, 'gui_icon.ico')
        )
        fn_script = os.path.join(pack_dir, 'installer.nsi')
        with open(os.path.join(simnibs_root_dir, 'packing', 'installer.nsi'), 'r') as f:
            install_script = Template(f.read()).render(
                version='.'.join(version.split('.')[:2]),
                full_version=version
            )
        with open(fn_script, 'w') as f:
            f.write(install_script)
        print('Creating NSIS installer')
        subprocess.run(
            fr'"%programfiles(x86)%\NSIS\makensis.exe" {fn_script}',
            check=True,
            shell=True
        )
        shutil.move(
            os.path.join(pack_dir, 'simnibs_installer_windows.exe'),
            'simnibs_installer_windows.exe'
        )
    if sys.platform == 'darwin':
        installer_name = "simnibs_installer_macos.pkg"
        with tempfile.TemporaryDirectory() as tmpdir:
            for fn in glob.glob(os.path.join(simnibs_root_dir, 'packing', 'macOS_installer', '*')):
                fn_out = os.path.join(tmpdir, os.path.basename(fn))
                with open(fn, 'r') as f:
                    template = Template(f.read()).render(
                        version='.'.join(version.split('.')[:2]),
                        full_version=version
                    )
                with open(fn_out, 'w') as f:
                    f.write(template)
                os.chmod(fn_out, os.stat(fn).st_mode)

            # Workaroud for Notarization
            # Instead of signing all binaries, I zip the enironment with a password
            # The postinstall script will unzip it in the user's computer
            print("Repacking...")
            orig_folder = os.path.abspath(os.curdir)
            os.chdir(pack_dir)
            subprocess.run([
                'zip', '-y', '-q', '-P', 'password', '-r',
                'simnibs_env.zip',
                'simnibs_env'
            ])
            os.chdir(orig_folder)
            shutil.rmtree(os.path.join(pack_dir, 'simnibs_env'))

            print('Running pkgbuild')
            subprocess.run([
                'pkgbuild',
                '--root', pack_dir,
                '--identifier', f'org.SimNIBS.{version}',
                '--version', version,
                '--scripts', tmpdir,
                '--install-location',
                '/Applications/SimNIBS-'+ '.'.join(version.split('.')[:2]),
                os.path.join(tmpdir, installer_name)
                ],
                check=True,
            )
            print('Running productbuild')
            if macos_developer_id is not None:
                sign = ['--sign', macos_developer_id]
            else:
                sign = []
            subprocess.run([
                'productbuild',
                '--distribution', os.path.join(tmpdir, 'Distribution'),
                '--package-path', tmpdir,
                '--resources', tmpdir,
                installer_name
                ] + sign,
                check=True
            )
    elif sys.platform=='linux':
        # Write the install script
        fn_script = os.path.join(pack_dir, 'install')
        with open(os.path.join(simnibs_root_dir, 'packing', 'install'), 'r') as f:
            install_script = Template(f.read()).render(
                version='.'.join(version.split('.')[:2]),
                full_version=version
            )
        with open(fn_script, 'w') as f:
            f.write(install_script)
        os.chmod(
            fn_script,
            os.stat(fn_script).st_mode |
            stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        print('Repacking...')
        installer_name = "simnibs_installer_linux"
        shutil.make_archive(
            installer_name,
            'gztar',
            # I use root_dir and base_dir so that it decompresses into a folder called
            # simnibs_installer
            root_dir='.',
            base_dir=os.path.relpath(pack_dir)
        )

    # Remove the (temporary) packing dir
    shutil.rmtree(pack_dir)

    # print(f"Created installer {installer_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="simnibs-pack",
        description="Create SimNIBS installers."
    )
    parser.add_argument("--env", default=None, help="Name of existing environment to use.")
    parser.add_argument("--simnibs-wheel-dir", default=None, help="Directory with the SimNIBS wheels to be packed")
    parser.add_argument("--macos-developer-id", default=None, help="Developer ID for signing in MacOS, DOES NOT SUPPORT NOTARIZATION (optional)")
    args = parser.parse_args(sys.argv[1:])
    build(args.env, args.simnibs_wheel_dir, args.macos_developer_id)
