import functools
import itertools
from multiprocessing import Pool
import numpy as np
import nibabel as nib
import nibabel.processing
import os
import scipy.sparse
from scipy.spatial import cKDTree, ConvexHull
from scipy.ndimage import label, binary_dilation
import time

from simnibs.mesh_tools import mesh_io
from simnibs.utils import file_finder
from simnibs.utils.simnibs_logger import logger
from simnibs.utils.spawn_process import spawn_process
from simnibs.utils.transformations import normalize

from brainsynth.dataset import GenericDataset
from brainnet.prediction import PretrainedModels
from brainnet.prediction.brainnet_predict import PredictionStep

from cortech import Surface, Hemisphere


class SimNIBSDataset(GenericDataset):
    def __init__(
        self,
        m2m_dirs: list | tuple[file_finder.SubjectFiles],
        image: str = "reference_volume",
        hemi: str | None = None,
    ):
        images = [getattr(m2m, image) for m2m in m2m_dirs]
        mni_transform = [m2m.coregistration_matrices for m2m in m2m_dirs]

        super().__init__(images, mni_transform, "mni2sub", "mni152", hemi)
        self.m2m_dirs = m2m_dirs

    def _load_mni_transform(self, index):
        # worldToWorldTransformMatrix is mni to subject space
        return scipy.io.loadmat(self.mni_transform[index])["worldToWorldTransformMatrix"]

    def preprocess_image(self, img):
        # Apply conform if the linear part of the affine deviates identity,
        # i.e., we want 1 mm voxels aligned the major axes in RAS orientation.
        if not np.allclose(img.affine[:3, :3], np.identity(3)):
            img = nibabel.processing.conform(img)
        return img


def cortical_surface_estimation(
        m2m_dirs: list | tuple[file_finder.SubjectFiles],
        model_name = "topofit",
        model_contrast="T1w",
        model_resolution="1mm",
        device="cpu"
    ):
    """
    Parameters
    ----------
    m2m_dirs


    Returns
    -------

    """
    dataset = SimNIBSDataset(m2m_dirs)

    name = model_name
    specs = (model_contrast.lower(), model_resolution.lower())
    pretrained_models = PretrainedModels()
    model = pretrained_models.load_model(name, specs, device)
    preprocessor = pretrained_models.load_preprocessor(name, specs, device)

    predict_step = PredictionStep(preprocessor, model, enable_amp=True)

    predictions = []
    for batch in dataset:
        y_pred, _ = predict_step(None, batch)
        y_pred = y_pred["surface"]

        hemispheres = {}
        for hemi, surfaces in y_pred.items():
            for surface,vertices in surfaces.items():
                y_pred[hemi][surface] = Surface(
                    vertices,
                    predict_step.topology[hemi].faces.cpu().numpy()
                )
            hemispheres[hemi] = Hemisphere(y_pred[hemi]["white"], y_pred[hemi]["pial"])
        predictions.append(hemispheres)
    return predictions


def central_surface_estimation(hemispheres, fraction=0.5, method="equivolume"):
    central = {}
    for k,v in hemispheres.items():
        thickness = v.compute_thickness()
        curv = v.compute_average_curvature(curv_kwargs=dict(smooth_iter=10))
        c = v.estimate_layers(thickness, curv.H, fraction, method)
        central[k] = Surface(c, v.white.faces)
    return central


# Spherical Registration

# def make_sphere_reg(hemispheres=None):
#     """This simply reads the spherical registration file from `deepsurfer` and
#     applies a correction to the right hemisphere."""
#     if hemispheres is None:
#         hemispheres = ("lh", "rh")

#     filename_sphere_reg = ds.system.resource('template/sphere-reg.srf')

#     sphere_reg = {}
#     sphere_reg["lh"] = mesh_io.read_freesurfer_surface(filename_sphere_reg)
#     # the mesh is the same for both hemispheres, only flipped in x
#     sphere_reg["rh"] = mesh_io.read_freesurfer_surface(filename_sphere_reg)
#     sphere_reg["rh"].nodes.node_coord *= (-1, 1, 1)

#     return sphere_reg


def spherical_registration_cat_parallel(m2m, n_processes: int = 2):
    """Run the spherical registration for left and right hemispheres in
    parallel.
    """
    fun = functools.partial(spherical_registration_cat, m2m)
    with Pool(processes=n_processes) as pool:
        pool.map(fun, m2m.hemispheres)


def spherical_registration_cat(m2m, hemi):
    """Compute spherical registration using CAT."""

    cat_surf2sphere = file_finder.path2bin("CAT_Surf2Sphere")
    cat_warpsurf = file_finder.path2bin("CAT_WarpSurf")

    white = m2m.get_surface(hemi, "white")
    sphere = m2m.surfaces["sphere"][hemi]
    sphere_reg = m2m.surfaces["sphere.reg"][hemi]

    fsavg_dir = file_finder.templates.freesurfer_templates
    fsavg_white = os.path.join(fsavg_dir, f'{hemi}.white.gii')
    fsavg_sphere = os.path.join(fsavg_dir, f'{hemi}.sphere.gii')

    # sphere creation (sphere)
    s = time.perf_counter()
    cmd = f"{cat_surf2sphere} {white} {sphere} 10"
    spawn_process(cmd.split())
    time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - s))
    print(f"Time for sphere generation ({hemi})      : {time_elapsed}")

    # registration to fsaverage (sphere.reg)
    s = time.perf_counter()
    cmd = f"{cat_warpsurf} -steps 2 -avg -i {white} -is {sphere} -t {fsavg_white} -ts {fsavg_sphere} -ws {sphere_reg}"
    spawn_process(cmd.split())
    time_elapsed = time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - s))
    print(f"Time for spherical registration ({hemi}) : {time_elapsed}")


def smooth_vertices(vertices, faces, verts2consider=None,
                    v2f_map=None, Niterations=1,
                    Ndilate=0, mask_move=None,
                    taubin=False):
    """Simple mesh smoothing by averaging vertex coordinates or other data
    across neighboring vertices.

    PARAMETERS
    ----------
    vertices : ndarray
        Vertices describing the mesh.
    faces : ndarray
        Faces describing the mesh.
    verts2consider: ndarray
        Array of indices of the vertex that will be smoothed (default: all vertices)
    v2f_map: {list, ndarray}
        Mapping from vertices to faces. Optional (to save a bit of time for repeated use),
        will be created if not given as input.
    Niterations: int
        Number of smoothing iterations (default: 1)
    Ndilate: int
        Number of times the surface region(s) defined by verts2consider are dilated
        before smoothing
    taubin: bool
        Wether to use Taubin smoothing. Defaut:False
    RETURNS
    ----------
    vertices : ndarray
        Vertices describing the expanded mesh.
    """

    if verts2consider is None:
        verts2consider = np.arange(len(vertices))
    if v2f_map is None:
        v2f_map = verts2faces(vertices,faces)

    for i in range(Ndilate):
        f2c = [v2f_map[n] for n in verts2consider]
        f2c, f2cok = list2numpy(f2c,dtype=int)
        f2c = f2c[f2cok]  # faces of verts2consider
        verts2consider = np.unique(faces[f2c])

    if mask_move is not None:
        verts2consider = verts2consider[mask_move[verts2consider]]

    smoo = vertices.copy()
    if taubin:
        m = mesh_io.Msh(nodes=mesh_io.Nodes(smoo),
                        elements=mesh_io.Elements(faces + 1))
        vert_mask = np.zeros(len(vertices), dtype=bool)
        vert_mask[verts2consider] = True
        m.smooth_surfaces_simple(Niterations, nodes_mask=vert_mask)
        smoo = m.nodes[:]
    else:
        for n in verts2consider:
            smoo[n] = np.average(vertices[faces[v2f_map[n]]], axis=(0,1))
        for i in range(Niterations-1):
            smoo2 = smoo.copy()
            for n in verts2consider:
                smoo[n] = np.average(smoo2[faces[v2f_map[n]]], axis=(0,1))
    return smoo



def get_element_neighbors(elements, ntol=1e-6):
    """Get the neighbors of each element in elements by comparing barycenters
    of element faces (e.g., if elements are tetrahedra, the faces are
    triangles).

    PARAMETERS
    ----------
    elements : ndarray
        Array of elements (e.g., triangles or tetrahedra) described as an
        N-by-M, with N being number of elements and M being the number of
        vertices of each element (e.g., 3 and 4 for triangles and tetrahedra,
        respectively).
    ntol : float, optional
        Neighbor tolerance. This parameters controls the upper bound for when
        elements are considered neighbors, i.e. the distance between elements
        has to be smaller than this value (default = 1e-6).

    RETURNS
    ----------
    nearest_neighbors : ndarray
        N-by-M array of indices into elements, i.e. for each face, which is its
        neighboring element.
    ok : ndarray (bool)
        This array tells, for each entry in nearest_neighbors, if this is an
        actual neighbor or not. The nearest neighbors are returned as a numpy
        ndarray of shape elements.shape for ease of interpretation and
        efficiency (and not for example as a list of lists of [possibly]
        unequal lengths), hence this is needed.
    """

    elements_idx = np.arange(len(elements))

    # barycenters of the faces making up each element
    barycenters = np.zeros_like(elements)
    num_nodes_per_el = elements.shape[1]
    for i in range(num_nodes_per_el):
        nodes = np.roll(np.arange(num_nodes_per_el),-i)[:-1] # nodes that make up the ith face
        barycenters[:,i,:] = np.average(elements[:,nodes,:], 1)

    bar_tree = cKDTree(barycenters.reshape(np.multiply(*elements.shape[:-1]),
                                           elements.shape[-1]))
    face_dist, face_idx = bar_tree.query(bar_tree.data, 2)

    nonself = (face_idx != np.arange(len(face_idx))[:,np.newaxis]) # get non-self-references

    # Distance to nearest neighbor. Neighbors having a distance shorter than
    # ntol are considered actual neighbors (i.e. sharing a face)
    face_dist = face_dist[nonself]
    ok = face_dist < ntol
    ok = ok.reshape(elements.shape[:2])

    # Index of nearest neigbor. From the tree search, indices are to nearest
    # element face, however, we wish to find neighboring elements. Hence,
    # reindex.
    face_idx = face_idx[nonself]
    nearest_neighbors = elements_idx.repeat(num_nodes_per_el)[face_idx]
    nearest_neighbors = nearest_neighbors.reshape(elements.shape[:2])

    return nearest_neighbors, ok



def verts2faces(vertices, faces, pad_val=0, array_out_type="list"):
    """Generate a mapping from vertices to faces in a mesh, i.e. for each
    vertices, which elements are it a part of.

    PARAMETERS
    ----------
    vertices : ndarray
        Vertices describing the mesh.
    faces : ndarray
        Faces describing the mesh.
    array_out_type : {"list", "numpy_array"}, optional
        Output type. Numpy arrays will enable vectorized operations to be
        performed on the output, however, in the case of variable number of
        elements per vertice, this will have to be padded

    RETURNS
    ----------
    v2f : {list, ndarray}
        The mapping from vertices to faces.
    ok : ndarray
        Array describing which entries in v2f are actual faces and which are
        "artificial". Since in a mesh, different vertices will often be part of
        different numbers of elements, some rows will have to be padded. This
        array is only returned if array_out_type is set to "numpy_array" since
        sublists of a list can be of variable length.
    """
    # Mapping from node to triangles, i.e. which nodes belongs to which
    # triangles
    v2f = [[] for i in range(len(vertices))]
    for t in range(len(faces)):
        for n in faces[t]:
            v2f[n].append(t)

    if array_out_type == "list":
        return v2f
    elif array_out_type == "numpy_array":
        v2f, ok = list2numpy(v2f, pad_val, int)
        return v2f, ok
    else:
        raise ValueError("Array output type must be list or numpy array.")



def list2numpy(L, pad_val=0, dtype=float):
    """Convert a python list of lists (the sublists being of varying length)
    to a numpy array.

    PARAMETERS
    ----------
    L : list
        The list of lists.
    pad_val : float, int
        The value with which to pad numpy array.
    dtype : datatype, optional
        Datatype of the output array.

    RETURNS
    ----------
    narr : ndarray
        L expressed as a numpy array.
    """

    max_neighbors = len(sorted(L, key=len, reverse=True)[0])
    narr = np.array([r+[np.nan]*(max_neighbors-len(r)) for r in L])
    ok = ~np.isnan(narr)
    narr[~ok] = pad_val
    narr = narr.astype(dtype)

    return narr, ok


def get_triangle_normals(mesh):
    """Get normal vectors for each triangle in the mesh.

    PARAMETERS
    ----------
    mesh : ndarray
        Array describing the surface mesh. The dimension are:
        [# of triangles] x [vertices (of triangle)] x [coordinates (of vertices)].

    RETURNS
    ----------
    tnormals : ndarray
        Normal vectors of each triangle in "mesh".
    """

    tnormals = np.cross(mesh[:,1,:]-mesh[:,0,:],mesh[:,2,:]-mesh[:,0,:]).astype(float)
    tnormals /= np.sqrt(np.sum(tnormals**2,1))[:,np.newaxis]
    return tnormals


def segment_triangle_intersect(vertices, faces, segment_start, segment_end):
    ''' Computes the intersection between a line segment and a triangulated surface

    Parameters
    -----------
    vertices: ndarray
        Array with mesh vertices positions
    faces: ndarray
        Array describing the surface triangles
    segment_start: ndarray
        N_lines x 2 array with the start of the line segments
    segment_end: ndarray
        N_lines x 2 array with the end of the line segments

    Returns
    --------
    indices_pairs: ndarray
        Nx2 array of ints with the pair (segment index, face index) for each intersection
    positions: ndarray
        Nx3 array of floats with the position of the intersections
    '''
    m = mesh_io.Msh(
            nodes=mesh_io.Nodes(vertices),
            elements=mesh_io.Elements(faces+1)
    )
    indices_pairs, positions = m.intersect_segment(segment_start, segment_end)
    # Go from 1-indexed to 0-indexed
    indices_pairs[:, 1] -= 1
    return indices_pairs, positions


def _rasterize_surface(vertices, faces, affine, shape, axis='z'):
    ''' Function to rastherize a given surface given by (vertices, faces) to a volume
    '''
    inv_affine = np.linalg.inv(affine)
    vertices_trafo = inv_affine[:3, :3].dot(vertices.T).T + inv_affine[:3, 3].T

    # switch vertices, dimensions to align with rastherization axis
    if axis == 'z':
        out_shape = shape
    elif axis == 'y':
        vertices_trafo = vertices_trafo[:, [0, 2, 1]]
        out_shape = np.array(shape, dtype=int)[[0, 2, 1]]
    elif axis == 'x':
        vertices_trafo = vertices_trafo[:, [2, 1, 0]]
        out_shape = np.array(shape, dtype=int)[[2, 1, 0]]
    else:
        raise ValueError('"axis" should be x, y, or z')

    grid_points = np.array(
        np.meshgrid(
            *tuple(map(np.arange, out_shape[:2])), indexing="ij"
        )
    ).reshape((2, -1)).T
    grid_points_near = np.hstack([grid_points, np.zeros((len(grid_points), 1))])
    grid_points_far = np.hstack([grid_points, out_shape[2] * np.ones((len(grid_points), 1))])

    # This fixes the search are such that if the volume area to rastherize is smaller
    # than the mesh, we will still trace rays that cross the whole extension of the mesh
    if np.min(vertices_trafo[:, 2]) < 0:
        grid_points_near[:, 2] = 1.1 * np.min(vertices_trafo[:, 2])
    if np.max(vertices_trafo[:, 2]) > out_shape[2]:
        grid_points_far[:, 2] = 1.1 * np.max(vertices_trafo[:, 2])

    # Calculate intersections
    pairs, positions = segment_triangle_intersect(
        vertices_trafo, faces, grid_points_near, grid_points_far
    )

    # Select the intersecting lines
    lines_intersecting, uq_indices, _, counts = np.unique(
        pairs[:, 0], return_index=True, return_inverse=True, return_counts=True
    )

    # The count should never be odd
    if np.any(counts % 2 == 1):
        logger.warning(
            'Found an odd number of crossings! This could be an open surface '
            'or a self-intersection'
        )

    # "z" voxels where intersections occurs
    #inter_z = np.around(positions[:, 2]).astype(int)
    inter_z = (positions[:, 2] + 1).astype(int)
    inter_z[inter_z < 0] = 0
    inter_z[inter_z > out_shape[2]] = out_shape[2]

    # needed to take care of last line
    uq_indices = np.append(uq_indices, [len(pairs)])

    # Go through each point in the grid and assign the z coordinates that are in the mesh
    # (between crossings)
    mask = np.zeros(out_shape, dtype=bool)
    for i, l in enumerate(lines_intersecting):
        # We can do this because we know that the "pairs" variables is ordered with
        # respect to the first variable
        crossings = np.sort(inter_z[uq_indices[i]: uq_indices[i+1]])
        for j in range(0, len(crossings) // 2):
            enter, leave = crossings[2*j], crossings[2*j + 1]
            mask[grid_points[l, 0], grid_points[l, 1], enter:leave] = True

    # Go back to the original frame
    if axis == 'z':
        pass
    elif axis == 'y':
        mask = np.swapaxes(mask, 2, 1)
    elif axis == 'x':
        mask = np.swapaxes(mask, 2, 0)

    return mask


def mask_from_surface(vertices, faces, affine, shape):
    """ Creates a binary mask based on a surface

    Parameters
    ----------
    vertices: ndarray
        Array with mesh vertices positions
    faces: ndarray
        Array describing the surface triangles
    affine: 4x4 ndarray
        Matrix describing the affine transformation between voxel and world coordinates
    shape: 3x1 list
        shape of output mask

    Returns
    ----------
    mask : ndarray of shape 'shape'
       Volume mask
    """

    masks = []

    if len(vertices) == 0 or len(faces) == 0:
        logger.warning("Surface if empty! Return empty volume")
        return np.zeros(shape, dtype=bool)

    # Do the rastherization in 3 directions
    for axis in ['x', 'y', 'z']:
        masks.append(_rasterize_surface(vertices, faces, affine, shape, axis=axis))

    # Return all voxels which are in at least 2 of the masks
    # This is done to reduce spurious results caused by bad tolopogy
    return np.sum(masks, axis=0) >= 2
    #return masks[2]


def dilate(image,n):
    nan_inds = np.isnan(image)
    image[nan_inds] = 0
    image = image > 0.5
    se = np.ones((2*n+1,2*n+1,2*n+1),dtype=bool)
    return binary_dilation(image,se)>0


def erosion(image,n):
    nan_inds = np.isnan(image)
    image[nan_inds] = 0
    image = image > 0.5
    return ~dilate(~image,n)


def lab(image):
    labels, _ = label(image)
    return (labels == np.argmax(np.bincount(labels.flat)[1:])+1)


def close(image,n):
    nan_inds = np.isnan(image)
    image[nan_inds] = 0
    image = image > 0.5
    image_padded = np.pad(image,n,'constant')
    image_padded = dilate(image_padded,n)
    image_padded = erosion(image_padded,n)
    return image_padded[n:-n,n:-n,n:-n]>0


def labclose(image,n):
    nan_inds = np.isnan(image)
    image[nan_inds] = 0
    image = image > 0.5
    tmp = close(image,n)
    return ~lab(~tmp)


def estimate_central_surface(white: mesh_io.Msh, pial: mesh_io.Msh):
    """For now, exploit the node-to-node correspondance between white and pial
    surfaces and just do an average to obtain the central surface.

    Parameters
    ----------
    white : mesh_io.Msh
    pial : mesh_io.Msh

    Returns
    -------
    mesh_io.Msh
        The estimated central surface.
    """
    vertices = 0.5 * (white.nodes.node_coord + pial.nodes.node_coord)
    return mesh_io.Msh(mesh_io.Nodes(vertices), white.elm)


def subsample_surfaces(m2m_dir, n_points: int) -> dict:
    """Subsample the surfaces files for each hemisphere (the original surfaces
    contain ~240,000 nodes per hemisphere).

    The subsampled surfaces are written to files in a subfolder in /surfaces,
    e.g., /surfaces/10000 for n_points=10000. All standard surfaces and morph
    data are subsampled. Additionally, the indices of the subsampled points in
    the original files are saved to {hemi}.index.csv and the normals from the
    full resolution surfaces corresponding to these nodes are written to a csv
    file {hemi}.normals.csv.

    PARAMETERS
    ----------
    m2m_dir : str
        Path to m2m subject folder.
    n_points : int
        Number of nodes in each subsampled hemisphere.

    RETURNS
    -------
    sub : dict
        Dictionary (hemispheres) of the subsampled central surface. Meshes
        contain fields 'index' (indices of points on original mesh) and
        'normals' (normals of points from original mesh).
    """
    logger.info(f"Downsampling brain surfaces to {n_points} points")

    m2m = file_finder.SubjectFiles(subpath=m2m_dir)

    full = dict(
        white = mesh_io.load_subject_surfaces(m2m, "white"),
        sphere = mesh_io.load_subject_surfaces(m2m, "sphere"),
    )
    subsampled = {h: subsample_surface(full["white"][h], full["sphere"][h], n_points) for h in full["white"]}

    # write subsampled central surface as well as index and normals
    for h, v in subsampled.items():
        filename = m2m.get_surface(h, "white", n_points)
        if not filename.parent.exists():
            filename.parent.mkdir()
        mesh_io.write_gifti_surface(v, filename)
        for name, data in v.field.items():
            filename = m2m.get_morph_data(h, name, n_points)
            filename = filename.with_suffix(f"{filename.suffix}.csv")
            if name == "index":
                np.savetxt(filename, data.value, "%i", ",")
            else:
                np.savetxt(filename, data.value, delimiter=",")

    # apply subsampling to all standard surfaces and morph data
    for s in m2m._standard_surfaces:
        if s == "white":
            continue
        m = mesh_io.load_subject_surfaces(m2m, s)
        for h, v in m.items():
            m = mesh_io.write_gifti_surface(
                mesh_io.Msh(
                    mesh_io.Nodes(v.nodes.node_coord[subsampled[h].field["index"].value]),
                    subsampled[h].elm,
                ),
                m2m.get_surface(h, s, n_points)
            )

    for d in m2m._standard_morph_data:
        data = mesh_io.load_subject_morph_data(m2m, d)
        for h, v in data.items():
            nib.freesurfer.write_morph_data(
                m2m.get_morph_data(h, d, n_points),
                v[subsampled[h].field["index"].value],
            )

    return subsampled


def subsample_surface(central_surf, sphere_surf, n_points, refine=True):
    """Subsample a hemisphere surface using its spherical registration.

    PARAMETERS
    ----------
    central_surf : dict
        Dictionary representing the surface of a hemisphere containing the keys
        "points" (points) and "tris" (triangulation).
    sphere_surf : dict
        Dictionary representing the surface of a hemisphere containing the keys
        "points" (points) and "tris" (triangulation).
    n_points : int
        Number of points (source positions) in the subsampled surface.

    RETURNS
    -------
    central_surf_sub : dict
        The subsampled surface. Also contains the key 'nn' representing the
        normals from the original surface.
    sphere_surf_sub : dict
        The subsampled surface.
    """
    assert isinstance(n_points, int)
    assert n_points < (n_full := central_surf.nodes.nr)

    # visualize = False # temporary debug flag for testing; requires pyvista

    # sphere_points = sphere_surf["points"] / np.linalg.norm(sphere_surf["points"], axis=1, keepdims=True)
    tree = cKDTree(normalize(sphere_surf.nodes.node_coord, axis=1))

    points, _ = fibonacci_sphere(n_points)
    _, idx = tree.query(points)

    # If multiple points map to the same points on the original surface then
    # keep only the unique ones.
    uniq = np.unique(idx)
    used = np.zeros(n_full, dtype=bool)
    used[uniq] = True

    n_smooth = get_n_smooth(n_points / n_full)
    basis = compute_gaussian_basis_functions(
        central_surf.nodes.node_coord,
        central_surf.elm.node_number_list[:, :3] - 1,
        n_smooth,
    ).tocsc()
    # rescale to avoid numerical problems?
    basis.data /= basis.mean(0).mean()
    coverage = np.asarray(
        basis[:, used].sum(1)
    ).ravel()  # i.e., basis @ x where x is indicator vector

    # if visualize:
    #     surfs = {}
    #     add_surfs(surfs, central_surf, sphere_surf, coverage.copy(), used, "init")

    # updates `coverage` in-place
    used, unused = maximize_coverage_by_addition(
        used, coverage, basis, n_points - uniq.size
    )
    # if visualize:
    #     add_surfs(surfs, central_surf, sphere_surf, coverage.copy(), used, "add")

    if refine:
        # updates `used`, `ununsed`, and `coverage` in-place
        indptr, indices, data = basis.indptr, basis.indices, basis.data
        equalize_coverage_by_swap(used, unused, coverage, indptr, indices, data)
        # covs, pairs, hard_swaps = equalize_coverage_by_swap(used, unused, coverage, indptr, indices, data)

    # Triangulate
    # hull = ConvexHull(sphere_surf['points'][used])
    hull = ConvexHull(sphere_surf.nodes.node_coord[used])
    points, tris = hull.points, hull.simplices
    ensure_orientation_consistency(points, tris)

    # if visualize:
    #     import pyvista as pv
    #     add_surfs(surfs, central_surf, sphere_surf, coverage.copy(), used, "swap")

    #     # visualize with pyvista
    #     clim = np.percentile(surfs["cent_init"]['coverage'], [5,95])
    #     names = ["init", "add", "swap"]

    #     p = pv.Plotter(shape=(2,3), window_size=(1600,1000), notebook=False)
    #     for i, name in enumerate(names):
    #         p.subplot(0,i)
    #         p.add_mesh(surfs[f"cent_{name}_sub"], show_edges=True)
    #         p.subplot(1,i)
    #         p.add_mesh(surfs[f"cent_{name}"], clim=clim, show_edges=True)
    #     # p.add_points(surfs[f'cent_{name}_sub'].points)
    #     p.link_views()
    #     p.show()

    # Use the normals from the original (high resolution) surface as this
    # should be more accurate
    idx = mesh_io.NodeData(used, "index")
    normals = mesh_io.NodeData(central_surf.nodes_normals().value[used], "normals")
    central_surf_sub = mesh_io.Msh(
        mesh_io.Nodes(central_surf.nodes.node_coord[used]),
        mesh_io.Elements(tris + 1),
    )
    # sphere_surf_sub = mesh_io.Msh(
    #     mesh_io.Nodes(sphere_surf.nodes.node_coord[used]),
    #     mesh_io.Elements(tris + 1),
    # )
    central_surf_sub.nodedata += [idx, normals]
    # sphere_surf_sub.nodedata += [idx]

    return central_surf_sub


def fibonacci_sphere(n, radius=1):
    """Generate a triangulated sphere with n vertices centered on (0, 0, 0).

    PARMETERS
    ---------
    n : int
        Number of vertices of the sphere.

    RETURNS
    -------
    rr : ndarray (n, 3)
        Point coordinates.
    tris : ndarray (m, 3)
        Array describing the triangulation.

    """
    points = fibonacci_sphere_points(n) * radius
    hull = ConvexHull(points)
    points, tris = hull.points, hull.simplices
    ensure_orientation_consistency(points, tris)
    return points, tris


def fibonacci_sphere_points(n):
    """Evenly distribute n points on a unit sphere using a Fibonacci lattice.

    PARAMETERS
    ----------
    n : int
        The desired number of points.

    RETURNS
    -------
    (n, 3) array with point coordinates in rows.

    NOTES
    -----
    Based on

        http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    """

    # Optimize average (instead of minimum) nearest neighbor distance by
    # introducing an offset at the poles
    epsilon = 0.36

    golden_ratio = 0.5 * (1 + np.sqrt(5))
    i = np.arange(0, n, dtype=float)

    # Original fibonacci lattice
    # x2, _ = np.modf(i / golden_ratio)
    # y2 = i / n_points

    # (MODIFIED) FIBONACCI LATTICE

    x2, _ = np.modf(i / golden_ratio)
    y2 = (i + epsilon) / (n - 1 + 2 * epsilon)

    # Project to fibonacci spiral via equal area projection
    # theta = 2 * np.pi * x2
    # r = np.sqrt(y2)

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(x2, y2)
    # ax.set_title('Fibonacci Lattice')

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='polar')
    # ax.scatter(theta, r)
    # ax.set_title('Fibonacci Spiral')

    # FIBONACCI SPHERE

    # Spherical coordinates (r = 1 is implicit because it is the unit sphere)
    # theta : longitude (around sphere, 0 <= theta <= 2*pi)
    # phi   : latitude (from pole to pole, 0 <= phi <= pi)
    theta = 2 * np.pi * x2
    phi = np.arccos(1 - 2 * y2)

    # Cartesian coordinates
    x3 = np.cos(theta) * np.sin(phi)
    y3 = np.sin(theta) * np.sin(phi)
    z3 = np.cos(phi)

    return np.array([x3, y3, z3]).T


def ensure_orientation_consistency(points, tris):
    """Fix orientation of normals so that they all point outwards. Operates
    in-place on tris.

    PARAMETERS
    ----------
    rr : ndarray (n, 3)
        Point coordinates.
    tris : ndarray (m, 3)
        Array describing the triangulation.

    RETURNS
    -------
    None, operates in-place on tris.
    """
    # centroid_tris: vector from global centroid to centroid of each triangle
    # (in this case the global centroid is [0,0,0] and so can be ignored)
    n = (
        mesh_io.Msh(mesh_io.Nodes(points), mesh_io.Elements(tris + 1))
        .triangle_normals()
        .value
    )
    centroid_tris = points[tris].mean(1)
    orientation = np.sum(centroid_tris * n, axis=1)
    swap_select_columns(tris, orientation < 0, [1, 2])


def swap_select_columns(arr, rows, cols):
    """Swap the columns (cols) of the selected rows (rows) in the array (arr).
    Operates in-place on arr.
    """
    assert not isinstance(rows, tuple)
    assert len(cols) == 2
    c0, c1 = cols
    arr[rows, c1], arr[rows, c0] = arr[rows, c0], arr[rows, c1]


def recursive_matmul(X, n):
    """Compute X @ X @ ... `n` times"""
    assert isinstance(n, int) and n >= 1
    return X if n == 1 else recursive_matmul(X, n - 1) @ X


def compute_adjacency_matrix(el, with_diag=False):
    """Make (sparse) adjacency matrix of vertices with connections as specified
    by `el`.

    PARAMETERS
    ----------
    el : ndarray
        n x m array describing the connectivity of the elements (e.g.,
        n x 3 for a triangulated of surface).
    with_diag : bool
        Include ones on the diagonal (default = False).

    RETURNS
    -------
    a : csr_matrix
        Sparse matrix in column order.
    """
    N = el.max() + 1

    pairs = np.array(list(itertools.combinations(np.arange(el.shape[1]), 2))).T
    row_ind = el[:, pairs.ravel()].ravel()
    col_ind = el[:, pairs[::-1].ravel()].ravel()

    data = np.ones_like(row_ind)
    a = scipy.sparse.csr_matrix((data / 2, (row_ind, col_ind)), shape=(N, N))

    if with_diag:
        a = a.tolil()
        a.setdiag(1)
        a = a.tocsr()

    return a


def compute_gaussian_basis_functions(points, tris, degree):
    """Generate a set of basis functions centered on each vertex"""
    A = compute_adjacency_matrix(tris)

    # Estimate standard deviation for Gaussian distance computation
    Acoo = A.tocoo()
    data = np.linalg.norm(points[Acoo.row] - points[Acoo.col], axis=1)
    # set sigma to half average distance to neighbors. This is arbitrary but
    # seems to work. Large sigmas do not seem to work well as they tend to
    # created a "striped" pattern in dense regions
    sigma = 0.5 * data.mean()

    # smooth to `neighborhood` degree neighbors
    A = recursive_matmul(A, degree)

    # compute gaussian distances
    Acoo = A.tocoo()
    data = np.linalg.norm(points[Acoo.row] - points[Acoo.col], axis=1)
    A.data = np.exp(-(data**2) / (2 * sigma**2))

    return A


def get_n_smooth(frac):
    """How much to smooth adjacency matrix. These numbers are heuristic at the
    moment.
    """
    n_smooth = 3
    if frac < 0.2:
        n_smooth += 1
    if frac < 0.05:
        n_smooth += 1
    return n_smooth


# @numba.jit(nopython=True, fastmath=True)
def update_vec(vec, indptr, indices, data, addi, subi):
    addi_indptr0, addi_indptr1 = indptr[addi : addi + 2]
    subi_indptr0, subi_indptr1 = indptr[subi : subi + 2]
    addi_indices = indices[addi_indptr0:addi_indptr1]
    subi_indices = indices[subi_indptr0:subi_indptr1]
    addi_data = data[addi_indptr0:addi_indptr1]
    subi_data = data[subi_indptr0:subi_indptr1]

    vec[addi_indices] += addi_data
    vec[subi_indices] -= subi_data


# @numba.jit(nopython=True, fastmath=True)
# def masked_indexed_argmin(x, index, mask, range_of_index):
#     """Equivalent to (but slightly faster than)

#         iargmin = range_of_index[mask][x[index[mask]].argmin()]
#         ixargmin = ixm[iargmin]

#     the more sparse/irregular `mask` is (since the boolean indexing operation
#     becomes slow).

#     """
#     iargmin, ixargmin = 0, 0
#     minval = np.inf
#     for i in range_of_index:
#         if mask[i]:
#             ixi = index[i]
#             val = x[ixi]
#             if val < minval:
#                 minval = val
#                 iargmin = i
#                 ixargmin = ixi
#     return iargmin, ixargmin, minval


# numba complains about the np.ones stuff. However, doesn't seem to make much
# difference
# @numba.jit(nopython=True, fastmath=True)
def maximize_coverage_by_addition(used, coverage, basis, n_add):
    """Add `n_add` points (move from unused to used) and update `coverage`
    accordingly (this is done in-place).

    PARAMTERS
    ---------
    used : ndarray of bool

    coverage : ndarray
        Array describing the coverage of the original points.

    n_add : int
        Number of points to add

    RETURNS
    -------
    used : ndarray
        Indices of the used points.
    unused : ndarray
        Indices of the unused points.
    """

    indptr, indices, data = basis.indptr, basis.indices, basis.data

    unused = np.where(~used)[0]
    added = -np.ones(n_add, dtype=int)
    mask = np.ones(unused.size, dtype=bool)
    range_of_index = np.arange(unused.size)

    for i in np.arange(n_add):
        # identify vertex to add
        # irem, iadd, _ = masked_indexed_argmin(coverage, unused, mask, range_of_index)
        irem = range_of_index[mask][coverage[unused[mask]].argmin()]
        iadd = unused[irem]

        mask[irem] = False
        added[i] = iadd

        # update coverage accordingly
        start, stop = indptr[iadd : iadd + 2]
        coverage[indices[start:stop]] += data[start:stop]

    used[added] = True
    unused = np.where(~used)[0]
    used = np.where(used)[0]

    return used, unused


# argpartition is not recognized by numba...
# @numba.jit(nopython=True, fastmath=True)
def equalize_coverage_by_swap(
    used, unused, coverage, indptr, indices, data, k=10, max_iter=5000
):
    """Try to equalize coverage by swapping used and unused points. In
    particular, the objective is to minimize the variance of the coverage.

    First, it tries to swap the points with minimum and maximum coverage (add
    and remove, respectively). If this does not decrease the variance, it tries
    the next k-1 pairs. If neither decrease the variance, the process
    terminates.

    Updates `used`, `unused`, and `coverage` in-place.

    PARAMTERS
    ---------
    used : ndarray
        Indices of the used points.
    unused : ndarray
        Indices of the unused points.
    coverage : ndarray
        Array describing the coverage of the original points.

    k : partition index
        Maximum number of pairs to try.


    """
    coverage_var = coverage.var()

    for _ in np.arange(max_iter):

        cov_un = coverage[unused]
        cov_us = coverage[used]
        addii = cov_un.argmin()
        subii = cov_us.argmax()
        addi = unused[addii]
        subi = used[subii]

        update_vec(coverage, indptr, indices, data, addi, subi)

        if (coverage_swap_var := coverage.var()) < coverage_var:
            unused[addii] = subi
            used[subii] = addi
            coverage_var = coverage_swap_var
        else:
            update_vec(coverage, indptr, indices, data, subi, addi)  # undo

            # this is the "slow" bit
            addiis = cov_un.argpartition(k)[:k]
            subiis = cov_us.argpartition(-k)[-k:]

            # order in pairs (skip the 1st as we already checked that)
            addiis = addiis[cov_un[addiis].argsort()][1:]
            subiis = subiis[cov_us[subiis].argsort()[::-1]][1:]

            found = False
            for addii, subii in zip(addiis, subiis):
                addi = unused[addii]
                subi = used[subii]

                update_vec(coverage, indptr, indices, data, addi, subi)

                if (coverage_swap_var := coverage.var()) < coverage_var:
                    found = True
                    unused[addii] = subi
                    used[subii] = addi
                    coverage_var = coverage_swap_var

                    break  # next iteration
                else:
                    update_vec(coverage, indptr, indices, data, subi, addi)  # undo

            if not found:
                break  # if nothing to swap, terminate


# some temporary stuff for plotting results of subsampling...
def add_surfs(surfs, central_surf, sphere_surf, coverage, used, name):
    import pyvista as pv

    surfs[f"cent_{name}"] = pv.make_tri_mesh(
        central_surf["points"], central_surf["tris"]
    )
    surfs[f"cent_{name}"]["coverage"] = coverage
    surfs[f"sphe_{name}"] = pv.make_tri_mesh(sphere_surf["points"], sphere_surf["tris"])
    surfs[f"sphe_{name}"]["coverage"] = coverage
    hull = ConvexHull(sphere_surf["points"][used])
    rr, tris = hull.points, hull.simplices
    ensure_orientation_consistency(rr, tris)
    surfs[f"cent_{name}_sub"] = pv.make_tri_mesh(central_surf["points"][used], tris)
    surfs[f"sphe_{name}_sub"] = pv.make_tri_mesh(sphere_surf["points"][used], tris)
