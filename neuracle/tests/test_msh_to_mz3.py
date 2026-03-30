import numpy as np

from neuracle.mesh_tools.msh_to_mz3 import _extract_surface_scalar_data
from simnibs import mesh_io
from simnibs.utils.mesh_element_properties import ElementTags


def _surface_mesh() -> mesh_io.Msh:
    nodes = mesh_io.Nodes(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    elements = mesh_io.Elements(
        triangles=np.array(
            [
                [1, 2, 3],
                [1, 3, 4],
            ]
        )
    )
    elements.tag1[:] = int(ElementTags.GM_TH_SURFACE)
    elements.tag2[:] = int(ElementTags.GM_TH_SURFACE)
    return mesh_io.Msh(nodes=nodes, elements=elements)


def test_extract_surface_scalar_data_from_surface_nodedata():
    mesh = _surface_mesh()
    mesh.add_node_field(np.array([10.0, 20.0, 30.0, 40.0]), "TImax")

    cdata = _extract_surface_scalar_data(mesh, mesh, "TImax")

    np.testing.assert_allclose(cdata[:, 0], [10.0, 20.0, 30.0, 40.0])


def test_extract_surface_scalar_data_from_surface_elementdata():
    mesh = _surface_mesh()
    mesh.add_element_field(np.array([1.0, 3.0]), "TImax")

    cdata = _extract_surface_scalar_data(mesh, mesh, "TImax")

    np.testing.assert_allclose(cdata[:, 0], [2.0, 1.0, 2.0, 3.0])
