"""
Build an interactive 3D Atlas - LEFT hemisphere only.
Full fsaverage resolution (163k vertices), no resampling.
Uses intensity + intensitymode='cell' for correct per-face hover.

Requirements
------------
    pip install nilearn mne nibabel plotly scipy

Data download
-------------
    On first run this script uses MNE to fetch the fsaverage template brain
    (~450 MB) from the Open Science Framework (OSF). The files are cached in
    ~/mne_data/MNE-fsaverage-data/ and are not re-downloaded on subsequent
    runs.  An internet connection is required for the initial download.
"""

import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import os
import sys
import json

FSAVERAGE_DIR = os.path.expanduser("~/mne_data/MNE-fsaverage-data/fsaverage")
SURF_DIR = os.path.join(FSAVERAGE_DIR, "surf")
LABEL_DIR = os.path.join(FSAVERAGE_DIR, "label")

# Atlas configurations
ATLAS_CONFIGS = {
    'glasser': {
        'annot_file': 'lh.HCPMMP1.annot',
        'name': 'Glasser Atlas',
        'description': 'HCP-MMP1.0 Parcellation',
        'citation': 'Glasser et al., 2016'
    },
    'destrieux': {
        'annot_file': 'lh.aparc.a2009s.annot',
        'name': 'Destrieux Atlas',
        'description': 'Destrieux Atlas (2009)',
        'citation': 'Destrieux et al., 2010'
    },
    'yeo7': {
        'annot_file': 'lh.Yeo2011_7Networks_N1000.annot',
        'name': 'Yeo 7 Networks',
        'description': 'Yeo 7 Resting-State Networks',
        'citation': 'Yeo et al., 2011'
    },
    'yeo17': {
        'annot_file': 'lh.Yeo2011_17Networks_N1000.annot',
        'name': 'Yeo 17 Networks',
        'description': 'Yeo 17 Resting-State Networks',
        'citation': 'Yeo et al., 2011'
    }
}

def load_surface(filepath):
    coords, faces = nib.freesurfer.read_geometry(filepath)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    return coords, faces

def clean_name(n, atlas_type):
    if atlas_type == 'glasser':
        n = n.replace('L_', '').replace('R_', '').replace('_ROI', '')
        if n == '???' or n == 'unknown':
            return 'Medial Wall'
    elif atlas_type == 'destrieux':
        # Clean Destrieux names
        n = n.replace('ctx-lh-', '').replace('ctx-rh-', '').replace('_', ' ')
        if n.startswith('Unknown'):
            return 'Unknown'
    elif atlas_type.startswith('yeo'):
        # Yeo networks are already clean
        pass
    return n

def build_atlas_data(atlas_type):
    config = ATLAS_CONFIGS[atlas_type]
    annot_file = config['annot_file']
    
    print(f"Loading full-res LH surfaces for {config['name']}...")
    lh_pial_v, lh_pial_f = load_surface(os.path.join(SURF_DIR, "lh.pial"))
    lh_infl_v, lh_infl_f = load_surface(os.path.join(SURF_DIR, "lh.inflated"))
    print(f"Vertices: {len(lh_pial_v)}, Faces: {len(lh_pial_f)}")

    print(f"Loading {config['name']} annotation...")
    lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(os.path.join(LABEL_DIR, annot_file))
    lh_names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in lh_names]
    print(f"Regions: {len(lh_names)}, Labels match vertices: {len(lh_labels) == len(lh_pial_v)}")

    # Per-face labels via majority vote
    print("Computing per-face labels...")
    n_faces = len(lh_pial_f)
    face_labels = np.zeros(n_faces, dtype=np.int32)
    for fi in range(n_faces):
        v0, v1, v2 = lh_pial_f[fi]
        l0, l1, l2 = lh_labels[v0], lh_labels[v1], lh_labels[v2]
        if l0 == l1 or l0 == l2:
            face_labels[fi] = l0
        else:
            face_labels[fi] = l1

    # Build discrete colorscale for intensity mapping
    n_regions = len(lh_names)
    colorscale = []
    for i in range(n_regions):
        r, g, b = lh_ctab[i, 0] / 255.0, lh_ctab[i, 1] / 255.0, lh_ctab[i, 2] / 255.0
        val = i / (n_regions - 1)
        colorscale.append([val, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])

    # Normalize face_labels to [0, 1] range matching colorscale
    face_intensity = face_labels.astype(np.float64) / (n_regions - 1)

    # Per-face hover text
    face_hover = [clean_name(lh_names[fl], atlas_type) if 0 <= fl < len(lh_names) else 'Unknown'
                  for fl in face_labels]

    # Find boundary edges between faces with different labels
    def find_boundary_edges(faces, face_labels_arr):
        from collections import defaultdict
        edge_faces = defaultdict(list)
        for fi, f in enumerate(faces):
            v0, v1, v2 = f
            for edge in tuple(sorted([v0,v1])), tuple(sorted([v1,v2])), tuple(sorted([v0,v2])):
                edge_faces[edge].append(fi)
        boundary = []
        for edge, flist in edge_faces.items():
            if len(flist) == 2 and face_labels_arr[flist[0]] != face_labels_arr[flist[1]]:
                boundary.append(edge)
        return boundary

    print("Finding boundary edges...")
    boundary = find_boundary_edges(lh_pial_f, face_labels)
    print(f"Boundary edges: {len(boundary)}")

    # Prepare data for JSON export
    data = {
        'atlas_type': atlas_type,
        'atlas_name': config['name'],
        'atlas_description': config['description'],
        'citation': config['citation'],
        'pv': [lh_pial_v[:, 0].tolist(), lh_pial_v[:, 1].tolist(), lh_pial_v[:, 2].tolist()],
        'iv': [lh_infl_v[:, 0].tolist(), lh_infl_v[:, 1].tolist(), lh_infl_v[:, 2].tolist()],
        'f': [lh_pial_f[:, 0].tolist(), lh_pial_f[:, 1].tolist(), lh_pial_f[:, 2].tolist()],
        'fi': face_intensity.tolist(),
        'cs': colorscale,
        'fn': lh_names,
        'fni': face_labels.tolist(),
        'pb': [[], [], []],  # boundary lines for pial
        'ib': [[], [], []]   # boundary lines for inflated
    }

    # Add boundary lines
    def add_boundary_lines(vertices, boundary_edges, target_list):
        xs, ys, zs = target_list
        for v0, v1 in boundary_edges:
            p0, p1 = vertices[v0], vertices[v1]
            off = 0.2
            n0 = p0 / (np.linalg.norm(p0) + 1e-10) * off
            n1 = p1 / (np.linalg.norm(p1) + 1e-10) * off
            xs.extend([p0[0]+n0[0], p1[0]+n1[0], None])
            ys.extend([p0[1]+n0[1], p1[1]+n1[1], None])
            zs.extend([p0[2]+n0[2], p1[2]+n1[2], None])

    add_boundary_lines(lh_pial_v, boundary, data['pb'])
    add_boundary_lines(lh_infl_v, boundary, data['ib'])

    return data

if __name__ == "__main__":
    atlas_type = sys.argv[1] if len(sys.argv) > 1 else 'glasser'
    
    if atlas_type not in ATLAS_CONFIGS:
        print(f"Available atlases: {list(ATLAS_CONFIGS.keys())}")
        sys.exit(1)
    
    data = build_atlas_data(atlas_type)
    
    output_file = f"../atlas_data_{atlas_type}.json"
    print(f"Writing data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print("Done!")
