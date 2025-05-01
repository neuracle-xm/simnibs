import argparse
from pathlib import Path
import sys
import warnings

import numpy as np

# import simnibs
from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.transformations import SurfaceMorph
from simnibs.mesh_tools import mesh_io
from simnibs.cli.utils import args_general


def main(m2m_dir, filename_mesh, out=None, fsaverage_res=None):
    """Map data from cortical surface vertices of a subject to fsaverage space.

    Parameters
    ----------
    m2m_dir : _type_
        The folder containing the CHARM run of the relevant subject. This
        folder contains the mapping from subject to fsaverage space.
    filename_mesh : _type_
        Filename of the mesh containing the data in subject space
    out : _type_, optional
        _description_, by default None
    fsaverage : _type_, optional
        _description_, by default None
    """
    # hardcoded but should probably be from mesh properties stuff...
    LH_TAG = 1
    RH_TAG = 2
    tag2hemi = {LH_TAG: "lh", RH_TAG: "rh"}

    m2m = SubjectFiles(subpath=m2m_dir)

    mesh = mesh_io.read_msh(filename_mesh)
    mesh = mesh.crop_mesh([1, 2])

    filename_mesh = Path(filename_mesh)
    out = (
        filename_mesh.parent / f"{filename_mesh.stem}_fsavg.msh"
        if out is None
        else Path(out)
    )
    if not out.parent.exists():
        out.mkdir()

    sub_sph = mesh_io.load_subject_surfaces(m2m, "sphere.reg")
    fsavg_sph = mesh_io.load_fsaverage_template("sphere", fsaverage_res)
    fsavg_surf = mesh_io.load_fsaverage_template("central", fsaverage_res)

    morph = {h: SurfaceMorph(sub_sph[h], fsavg_sph[h]) for h in m2m.hemispheres}

    # indices of vertices per hemisphere
    idx = {
        hemi: np.unique(mesh.elm.node_number_list[mesh.elm.tag1 == tag, :3] - 1)
        for tag, hemi in tag2hemi.items()
    }

    values = []
    names = []
    for nodedata in mesh.nodedata:  # Only node data is supported
        data = np.concatenate(
            [
                morph[hemi].transform(nodedata.value[idx[hemi]])
                for hemi in tag2hemi.values()
            ]
        )
        values.append(data)
        names.append(nodedata.field_name)
    if len(mesh.elmdata) > 1:
        warnings.warn("Element data present in mesh file. This will not be mapped to fsaverage space.")

    tags = np.concatenate(
        (
            np.full_like(fsavg_surf["lh"].elm.tag1, LH_TAG),
            np.full_like(fsavg_surf["rh"].elm.tag1, RH_TAG),
        )
    )
    merged = fsavg_surf["lh"].join_mesh(fsavg_surf["rh"])
    merged.elm.tag1 = tags
    merged.elm.tag2 = tags
    for name, value in zip(names, values):
        merged.add_node_field(value, name)

    merged.write(str(out))


def parse_args(argv):
    description = "Map data defined on mesh nodes from subject space to fsaverage space. This function requires an existing CHARM run (m2m directory) for the subject on whose cortical surface(s) the data is defined."
    usage = "subject2fsaverage /path/to/m2m_subID /path/to/mesh.msh"
    parser = argparse.ArgumentParser(
        "subject2fsaverage",
        usage,
        description,
        # formatter_class=argparse.RawTextHelpFormatter
    )
    args_general.subid.add_to(parser)
    parser.add_argument(
        "mesh", help="Name of a mesh file containing data defined on the mesh nodes to map to fsaverage (e.g., optimization results)."
    )
    parser.add_argument(
        "--out",
        help="The filename to save the results to. If not specified will append '_fsavg' to the filename of the input.",
    )
    args_general.fsaverage.add_to(parser)

    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)

    main(args.subid, args.mesh, args.out, args.fsaverage)
