"""
build.py

Turns a VertexLabels result (from any AtlasSource.build()) into the
JSON structure your Plotly viewer consumes: surface geometry, per-face
labels/colours, and parcel boundary lines. This logic is identical
regardless of where the labels came from (.annot, volumetric, custom
URL), which is the point of routing everything through VertexLabels.
"""

import os
from collections import defaultdict

import numpy as np
import nibabel as nib

from .core import AtlasSource, ensure_fsaverage_full, ensure_fsaverage6_surfaces

N_FSAVERAGE6 = 40962  # vertices per hemisphere


def load_surface(filepath):
    """Load a surface mesh regardless of format (FreeSurfer binary or
    GIFTI). Detects by content rather than file extension, since
    nilearn's fetched file names don't reliably carry a '.gii' suffix
    across versions."""
    filepath = str(filepath)
    try:
        coords, faces = nib.freesurfer.read_geometry(filepath)
    except ValueError:
        gii = nib.load(filepath)
        coords = gii.darrays[0].data
        faces = gii.darrays[1].data
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    return coords, faces


def _load_lh_surfaces(resolution):
    """Return (pial_v, pial_f, infl_v, infl_f) for the requested
    resolution ('fsaverage' or 'fsaverage6')."""
    if resolution == "fsaverage6":
        fsavg6 = ensure_fsaverage6_surfaces()
        pial_v, pial_f = load_surface(fsavg6["pial_left"])
        infl_v, infl_f = load_surface(fsavg6["infl_left"])
        return pial_v, pial_f, infl_v, infl_f

    fs_dir = ensure_fsaverage_full()
    surf_dir = os.path.join(fs_dir, "surf")
    pial_v, pial_f = load_surface(os.path.join(surf_dir, "lh.pial"))
    infl_v, infl_f = load_surface(os.path.join(surf_dir, "lh.inflated"))
    return pial_v, pial_f, infl_v, infl_f


def clean_name(n, atlas_key):
    if "HCPMMP1" in atlas_key:
        n = n.replace("L_", "").replace("R_", "").replace("_ROI", "")
        if n in ("???", "unknown"):
            return "Medial Wall"
    elif "aparc" in atlas_key:
        n = n.replace("ctx-lh-", "").replace("ctx-rh-", "").replace("_", " ")
        if n.startswith("Unknown"):
            return "Unknown"
    return n


def _face_majority_labels(faces, vertex_labels):
    n_faces = len(faces)
    face_labels = np.zeros(n_faces, dtype=np.int32)
    for fi in range(n_faces):
        v0, v1, v2 = faces[fi]
        l0, l1, l2 = vertex_labels[v0], vertex_labels[v1], vertex_labels[v2]
        face_labels[fi] = l0 if (l0 == l1 or l0 == l2) else l1
    return face_labels


def _find_boundary_edges(faces, face_labels_arr):
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        v0, v1, v2 = f
        for edge in (tuple(sorted([v0, v1])),
                     tuple(sorted([v1, v2])),
                     tuple(sorted([v0, v2]))):
            edge_faces[edge].append(fi)
    boundary = []
    for edge, flist in edge_faces.items():
        if len(flist) == 2 and face_labels_arr[flist[0]] != face_labels_arr[flist[1]]:
            boundary.append(edge)
    return boundary


def _add_boundary_lines(vertices, boundary_edges, target_list):
    xs, ys, zs = target_list
    for v0, v1 in boundary_edges:
        p0, p1 = vertices[v0], vertices[v1]
        off = 0.2
        n0 = p0 / (np.linalg.norm(p0) + 1e-10) * off
        n1 = p1 / (np.linalg.norm(p1) + 1e-10) * off
        xs.extend([p0[0] + n0[0], p1[0] + n1[0], None])
        ys.extend([p0[1] + n0[1], p1[1] + n1[1], None])
        zs.extend([p0[2] + n0[2], p1[2] + n1[2], None])


def _generate_categorical_ctab(n_regions):
    """Fallback colour table for atlases with no ctab (e.g. most
    volumetric nilearn atlases).

    Uses HSV with a golden-angle hue step rather than cycling a fixed
    20-colour palette: tab20 (or any small qualitative colormap) repeats
    every 20 regions, so region 21 gets the same colour as region 1 -
    fine for Desikan-Killiany (~36 regions) but actively misleading for
    atlases with hundreds of regions (Schaefer-400, HCP-MMP1's 180).
    The golden angle (~137.5 deg) spaces hues so consecutive regions
    never land near each other in hue, and saturation/value are varied
    across passes around the hue wheel so even same-hue regions from
    different passes are still visually distinct.
    """
    import colorsys

    golden_angle = 0.6180339887498949  # 1/phi, in units of a full hue turn
    ctab = np.zeros((n_regions, 4), dtype=np.int32)
    for i in range(n_regions):
        hue = (i * golden_angle) % 1.0
        # Vary saturation/value slowly so colours from later "wraps"
        # around the hue wheel are still distinguishable from earlier
        # ones at a similar hue.
        sat = 0.55 + 0.35 * (((i * 0.37) % 1.0))
        val = 0.75 + 0.2 * (((i * 0.53) % 1.0))
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        ctab[i] = [int(r * 255), int(g * 255), int(b * 255), 255]
    return ctab


def build_atlas_json(atlas_source: AtlasSource, surf_dir=None):
    """Run an AtlasSource end-to-end and return the JSON-ready dict.

    If atlas_source.resolution == 'fsaverage6', the fsaverage6 surface
    mesh is used and the (always full-resolution) vertex labels from
    the transformer are sliced down to the first 40,962 vertices, which
    correspond 1:1 to the same anatomical locations thanks to FreeSurfer's
    nested icosahedral subdivision (fsaverage6 vertex i == fsaverage
    vertex i, for i < 40,962). Verify this once against a real FreeSurfer
    install if exact vertex correspondence matters for your use case.
    """
    resolution = atlas_source.resolution

    print(f"Fetching + transforming {atlas_source.name}...")
    vertex_labels = atlas_source.build()

    if surf_dir is not None:
        # explicit override - always full-res in this case
        lh_pial_v, lh_pial_f = load_surface(os.path.join(surf_dir, "lh.pial"))
        lh_infl_v, lh_infl_f = load_surface(os.path.join(surf_dir, "lh.inflated"))
    else:
        print(f"Loading LH surfaces ({resolution})...")
        lh_pial_v, lh_pial_f, lh_infl_v, lh_infl_f = _load_lh_surfaces(resolution)
    print(f"Vertices: {len(lh_pial_v)}, Faces: {len(lh_pial_f)}")

    lh_labels = vertex_labels.labels
    lh_names = vertex_labels.names
    n_regions = len(lh_names)

    if resolution == "fsaverage6" and len(lh_labels) != len(lh_pial_v):
        if len(lh_labels) < len(lh_pial_v):
            raise ValueError(
                f"{atlas_source.name}: label array has {len(lh_labels)} "
                f"vertices, fewer than the {len(lh_pial_v)}-vertex "
                f"fsaverage6 mesh - can't slice down further."
            )
        print(f"Slicing labels from {len(lh_labels)} (full-res) down to "
              f"{len(lh_pial_v)} (fsaverage6) vertices...")
        lh_labels = lh_labels[:len(lh_pial_v)]

    print(f"Regions: {n_regions}, Labels match vertices: {len(lh_labels) == len(lh_pial_v)}")

    print("Computing per-face labels...")
    face_labels = _face_majority_labels(lh_pial_f, lh_labels)

    ctab = vertex_labels.ctab
    if ctab is None:
        ctab = _generate_categorical_ctab(n_regions)

    # Per-face explicit RGB colour strings, rather than an
    # intensity+colorscale pair. plotly.js's colorscale interpolation
    # breaks for colorscales with >=256 stops ("map requires nshades to
    # be at least size N" - see plotly/plotly.js#3699), which atlases
    # with hundreds of regions (e.g. Schaefer-400) blow straight past.
    # facecolor sidesteps that entirely: no interpolation, just direct
    # per-face colour lookup, and it works the same regardless of how
    # many regions an atlas has.
    face_rgb = ctab[np.clip(face_labels, 0, len(ctab) - 1), :3]
    face_colors = [f"rgb({r},{g},{b})" for r, g, b in face_rgb.tolist()]

    print("Finding boundary edges...")
    boundary = _find_boundary_edges(lh_pial_f, face_labels)
    print(f"Boundary edges: {len(boundary)}")

    data = {
        "atlas_key": atlas_source.key,
        "atlas_name": atlas_source.name,
        "atlas_description": atlas_source.description,
        "citation": atlas_source.citation,
        "license_note": atlas_source.license_note,
        "pv": [lh_pial_v[:, 0].tolist(), lh_pial_v[:, 1].tolist(), lh_pial_v[:, 2].tolist()],
        "iv": [lh_infl_v[:, 0].tolist(), lh_infl_v[:, 1].tolist(), lh_infl_v[:, 2].tolist()],
        "f": [lh_pial_f[:, 0].tolist(), lh_pial_f[:, 1].tolist(), lh_pial_f[:, 2].tolist()],
        "fc": face_colors,
        "fn": lh_names,
        "fni": face_labels.tolist(),
        "pb": [[], [], []],
        "ib": [[], [], []],
    }
    _add_boundary_lines(lh_pial_v, boundary, data["pb"])
    _add_boundary_lines(lh_infl_v, boundary, data["ib"])

    return data