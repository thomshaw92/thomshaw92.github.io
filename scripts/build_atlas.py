"""
Build interactive 3D Atlases - LEFT hemisphere only.
Full fsaverage resolution (163k vertices), no resampling.
Uses intensity + intensitymode='cell' for correct per-face hover.

Supports all MNE/FreeSurfer atlases and custom atlas downloads.

Requirements
------------
    pip install nilearn mne nibabel plotly scipy requests

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
import glob
import requests
from urllib.parse import urlparse
import tempfile

FSAVERAGE_DIR = os.path.expanduser("~/mne_data/MNE-fsaverage-data/fsaverage")
SURF_DIR = os.path.join(FSAVERAGE_DIR, "surf")
LABEL_DIR = os.path.join(FSAVERAGE_DIR, "label")

# Known atlas metadata
ATLAS_METADATA = {
    'lh.HCPMMP1.annot': {
        'name': 'Glasser Atlas (HCP-MMP1.0)',
        'description': 'HCP-MMP1.0 Parcellation',
        'citation': 'Glasser et al., 2016'
    },
    'lh.aparc.a2009s.annot': {
        'name': 'Destrieux Atlas',
        'description': 'Destrieux Atlas (2009)',
        'citation': 'Destrieux et al., 2010'
    },
    'lh.aparc.annot': {
        'name': 'Desikan-Killiany Atlas',
        'description': 'Desikan-Killiany Atlas (2006)',
        'citation': 'Desikan et al., 2006'
    },
    'lh.Yeo2011_7Networks_N1000.annot': {
        'name': 'Yeo 7 Networks',
        'description': 'Yeo 7 Resting-State Networks',
        'citation': 'Yeo et al., 2011'
    },
    'lh.Yeo2011_17Networks_N1000.annot': {
        'name': 'Yeo 17 Networks',
        'description': 'Yeo 17 Resting-State Networks',
        'citation': 'Yeo et al., 2011'
    },
    'lh.PALS_B12_Lobes.annot': {
        'name': 'PALS Lobe Atlas',
        'description': 'PALS Lobe Parcellation',
        'citation': 'Van Essen, 2005'
    },
    'lh.PALS_B12_Brodmann.annot': {
        'name': 'PALS Brodmann Atlas',
        'description': 'PALS Brodmann Areas',
        'citation': 'Van Essen, 2005'
    },
    'lh.aparc.a2005s.annot': {
        'name': 'Destrieux 2005 Atlas',
        'description': 'Destrieux Atlas (2005)',
        'citation': 'Destrieux et al., 2010'
    },
    'lh.oasis.chubs.annot': {
        'name': 'OASIS CHUBS Atlas',
        'description': 'OASIS CHUBS Parcellation',
        'citation': 'OASIS'
    }
}

def load_surface(filepath):
    coords, faces = nib.freesurfer.read_geometry(filepath)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    return coords, faces

def clean_name(n, atlas_filename):
    atlas_type = atlas_filename.replace('lh.', '').replace('.annot', '')
    
    if 'HCPMMP1' in atlas_type:
        n = n.replace('L_', '').replace('R_', '').replace('_ROI', '')
        if n == '???' or n == 'unknown':
            return 'Medial Wall'
    elif 'aparc' in atlas_type:
        # Clean aparc names
        n = n.replace('ctx-lh-', '').replace('ctx-rh-', '').replace('_', ' ')
        if n.startswith('Unknown'):
            return 'Unknown'
    elif 'Yeo' in atlas_type:
        # Yeo networks are already clean
        pass
    elif 'PALS' in atlas_type:
        # PALS names are usually clean
        pass
    return n

def download_custom_atlas(url, temp_dir):
    """Download a custom atlas annotation file."""
    response = requests.get(url)
    response.raise_for_status()
    
    # Determine filename from URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename.endswith('.annot'):
        filename += '.annot'
    
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    return filepath, filename

def process_atlas(annot_file, atlas_name=None, atlas_description=None, citation=None):
    """Process a single atlas annotation file."""
    
    print(f"Loading full-res LH surfaces for {atlas_name or os.path.basename(annot_file)}...")
    lh_pial_v, lh_pial_f = load_surface(os.path.join(SURF_DIR, "lh.pial"))
    lh_infl_v, lh_infl_f = load_surface(os.path.join(SURF_DIR, "lh.inflated"))
    print(f"Vertices: {len(lh_pial_v)}, Faces: {len(lh_pial_f)}")

    print(f"Loading {atlas_name or os.path.basename(annot_file)} annotation...")
    lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(annot_file)
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
    face_hover = [clean_name(lh_names[fl], os.path.basename(annot_file)) if 0 <= fl < len(lh_names) else 'Unknown'
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
    atlas_key = os.path.basename(annot_file).replace('lh.', '').replace('.annot', '')
    data = {
        'atlas_key': atlas_key,
        'atlas_filename': os.path.basename(annot_file),
        'atlas_name': atlas_name or atlas_key,
        'atlas_description': atlas_description or f'{atlas_key} atlas',
        'citation': citation or 'Unknown',
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

def discover_available_atlases():
    """Discover all available atlas annotation files."""
    pattern = os.path.join(LABEL_DIR, "lh.*.annot")
    annot_files = glob.glob(pattern)
    return sorted(annot_files)

def build_atlas_registry():
    """Build a registry of all available atlases."""
    registry = {}
    
    # Add discovered atlases
    for annot_file in discover_available_atlases():
        filename = os.path.basename(annot_file)
        if filename in ATLAS_METADATA:
            metadata = ATLAS_METADATA[filename]
        else:
            # Generic metadata for unknown atlases
            atlas_key = filename.replace('lh.', '').replace('.annot', '')
            metadata = {
                'name': atlas_key.replace('_', ' ').title(),
                'description': f'{atlas_key} atlas',
                'citation': 'Unknown'
            }
        
        registry[filename] = metadata
    
    return registry

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python build_glasser_atlas.py list                    # List all available atlases")
        print("  python build_glasser_atlas.py build-all               # Build all available atlases")
        print("  python build_glasser_atlas.py build <atlas_key>       # Build specific atlas")
        print("  python build_glasser_atlas.py custom <url> [name] [description] [citation]  # Download and build custom atlas")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'list':
        print("Available atlases:")
        registry = build_atlas_registry()
        for filename, metadata in registry.items():
            atlas_key = filename.replace('lh.', '').replace('.annot', '')
            print(f"  {atlas_key}: {metadata['name']} - {metadata['description']}")
    
    elif command == 'build-all':
        registry = build_atlas_registry()
        for filename, metadata in registry.items():
            atlas_key = filename.replace('lh.', '').replace('.annot', '')
            print(f"\nBuilding {metadata['name']}...")
            try:
                data = process_atlas(
                    os.path.join(LABEL_DIR, filename),
                    metadata['name'],
                    metadata['description'],
                    metadata['citation']
                )
                output_file = f"../atlas_data_{atlas_key}.json"
                print(f"Writing data to {output_file}...")
                with open(output_file, 'w') as f:
                    json.dump(data, f)
                print("Done!")
            except Exception as e:
                print(f"Failed to build {metadata['name']}: {e}")
    
    elif command == 'build':
        if len(sys.argv) < 3:
            print("Usage: python build_glasser_atlas.py build <atlas_key>")
            sys.exit(1)
        
        atlas_key = sys.argv[2]
        filename = f"lh.{atlas_key}.annot"
        annot_file = os.path.join(LABEL_DIR, filename)
        
        if not os.path.exists(annot_file):
            print(f"Atlas {atlas_key} not found. Available atlases:")
            registry = build_atlas_registry()
            for fname in registry.keys():
                key = fname.replace('lh.', '').replace('.annot', '')
                print(f"  {key}")
            sys.exit(1)
        
        registry = build_atlas_registry()
        metadata = registry.get(filename, {
            'name': atlas_key.replace('_', ' ').title(),
            'description': f'{atlas_key} atlas',
            'citation': 'Unknown'
        })
        
        data = process_atlas(
            annot_file,
            metadata['name'],
            metadata['description'],
            metadata['citation']
        )
        
        output_file = f"../atlas_data_{atlas_key}.json"
        print(f"Writing data to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(data, f)
        print("Done!")
    
    elif command == 'custom':
        if len(sys.argv) < 3:
            print("Usage: python build_glasser_atlas.py custom <url> [name] [description] [citation]")
            sys.exit(1)
        
        url = sys.argv[2]
        name = sys.argv[3] if len(sys.argv) > 3 else None
        description = sys.argv[4] if len(sys.argv) > 4 else None
        citation = sys.argv[5] if len(sys.argv) > 5 else None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                annot_file, filename = download_custom_atlas(url, temp_dir)
                print(f"Downloaded custom atlas to {annot_file}")
                
                data = process_atlas(annot_file, name, description, citation)
                
                # Use provided name or derive from filename
                atlas_key = name.replace(' ', '_').lower() if name else filename.replace('.annot', '')
                output_file = f"../atlas_data_{atlas_key}.json"
                print(f"Writing data to {output_file}...")
                with open(output_file, 'w') as f:
                    json.dump(data, f)
                print("Done!")
                
            except Exception as e:
                print(f"Failed to process custom atlas: {e}")
                sys.exit(1)
    
    else:
        print("Unknown command. Use 'list', 'build-all', 'build <atlas_key>', or 'custom <url>'")
        sys.exit(1)