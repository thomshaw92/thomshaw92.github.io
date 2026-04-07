"""
Build an interactive 3D Glasser Atlas - LEFT hemisphere only.
Full fsaverage resolution (163k vertices), no resampling.
Uses intensity + intensitymode='cell' for correct per-face hover.
"""

import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import os

FSAVERAGE_DIR = os.path.expanduser("~/mne_data/MNE-fsaverage-data/fsaverage")
SURF_DIR = os.path.join(FSAVERAGE_DIR, "surf")
LABEL_DIR = os.path.join(FSAVERAGE_DIR, "label")

def load_surface(filepath):
    coords, faces = nib.freesurfer.read_geometry(filepath)
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    return coords, faces

print("Loading full-res LH surfaces...")
lh_pial_v, lh_pial_f = load_surface(os.path.join(SURF_DIR, "lh.pial"))
lh_infl_v, lh_infl_f = load_surface(os.path.join(SURF_DIR, "lh.inflated"))
print(f"Vertices: {len(lh_pial_v)}, Faces: {len(lh_pial_f)}")

print("Loading Glasser annotation...")
lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(os.path.join(LABEL_DIR, "lh.HCPMMP1.annot"))
lh_names = [n.decode('utf-8') if isinstance(n, bytes) else n for n in lh_names]
print(f"Regions: {len(lh_names)}, Labels match vertices: {len(lh_labels) == len(lh_pial_v)}")

def clean_name(n):
    n = n.replace('L_', '').replace('R_', '').replace('_ROI', '')
    if n == '???' or n == 'unknown':
        return 'Medial Wall'
    return n

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
# intensity will be the face_label index (0..180)
# colorscale maps each value to its color
n_regions = len(lh_names)
colorscale = []
for i in range(n_regions):
    r, g, b = lh_ctab[i, 0] / 255.0, lh_ctab[i, 1] / 255.0, lh_ctab[i, 2] / 255.0
    val = i / (n_regions - 1)
    colorscale.append([val, f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])

# Normalize face_labels to [0, 1] range matching colorscale
face_intensity = face_labels.astype(np.float64) / (n_regions - 1)

# Per-face hover text
face_hover = [clean_name(lh_names[fl]) if 0 <= fl < len(lh_names) else 'Unknown'
              for fl in face_labels]

# Find boundary edges between faces with different labels
def find_boundary_edges(faces, face_labels_arr):
    from collections import defaultdict
    edge_faces = defaultdict(list)
    for fi, f in enumerate(faces):
        v0, v1, v2 = f
        for edge in [tuple(sorted([v0,v1])), tuple(sorted([v1,v2])), tuple(sorted([v0,v2]))]:
            edge_faces[edge].append(fi)
    boundary = []
    for edge, flist in edge_faces.items():
        if len(flist) == 2 and face_labels_arr[flist[0]] != face_labels_arr[flist[1]]:
            boundary.append(edge)
    return boundary

print("Finding boundary edges...")
boundary = find_boundary_edges(lh_pial_f, face_labels)
print(f"Boundary edges: {len(boundary)}")

def make_boundary_lines(vertices, boundary_edges, name, visible=True):
    xs, ys, zs = [], [], []
    for v0, v1 in boundary_edges:
        p0, p1 = vertices[v0], vertices[v1]
        off = 0.2
        n0 = p0 / (np.linalg.norm(p0) + 1e-10) * off
        n1 = p1 / (np.linalg.norm(p1) + 1e-10) * off
        xs.extend([p0[0]+n0[0], p1[0]+n1[0], None])
        ys.extend([p0[1]+n0[1], p1[1]+n1[1], None])
        zs.extend([p0[2]+n0[2], p1[2]+n1[2], None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color='black', width=3),
        name=name, visible=visible,
        hoverinfo='skip', showlegend=False,
    )

def make_mesh(vertices, faces, face_intensity_arr, face_hover_arr, cscale, name, visible=True):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=face_intensity_arr,
        intensitymode='cell',
        colorscale=cscale,
        showscale=False,
        name=name,
        visible=visible,
        # Per-face hover text
        text=face_hover_arr,
        hovertemplate='%{text}<extra></extra>',
        flatshading=False,
        lighting=dict(ambient=0.8, diffuse=0.5, specular=0.05, roughness=0.8, fresnel=0.0),
        lightposition=dict(x=0, y=-300, z=300),
    )

print("Building traces...")
traces = [
    make_mesh(lh_pial_v, lh_pial_f, face_intensity, face_hover, colorscale, "LH Pial", visible=True),
    make_boundary_lines(lh_pial_v, boundary, "Pial Borders", visible=True),
    make_mesh(lh_infl_v, lh_infl_f, face_intensity, face_hover, colorscale, "LH Inflated", visible=False),
    make_boundary_lines(lh_infl_v, boundary, "Infl Borders", visible=False),
]

pial_vis = [True, True, False, False]
infl_vis = [False, False, True, True]

fig = go.Figure(data=traces)

fig.update_layout(
    title=dict(
        text="<b>Interactive Glasser Atlas</b><br><span style='font-size:13px;color:#aaa'>HCP-MMP1.0 Parcellation (Left Hemisphere) | Glasser et al., 2016</span>",
        font=dict(size=22, color='white'),
        x=0.5, y=0.98, yanchor='top',
    ),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='#161629',
        aspectmode='data',
        camera=dict(
            eye=dict(x=-1.7, y=-0.5, z=0.3),
            up=dict(x=0, y=0, z=1)
        )
    ),
    paper_bgcolor='#161629',
    plot_bgcolor='#161629',
    font=dict(color='white', family='Inter, Helvetica, Arial, sans-serif'),
    margin=dict(l=0, r=0, t=110, b=40),
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            x=0.98, xanchor="right",
            y=0.85, yanchor="top",
            buttons=[
                dict(label="  Folded (Pial)  ", method="update", args=[{"visible": pial_vis}]),
                dict(label="  Inflated  ", method="update", args=[{"visible": infl_vis}]),
            ],
            bgcolor='#2a2a44',
            font=dict(color='white', size=13),
            bordercolor='#4a4a6a', borderwidth=1,
        )
    ],
    annotations=[
        dict(
            text="Hover to see region names | Click+drag to rotate | Scroll to zoom",
            x=0.5, y=-0.01, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color='#666')
        )
    ]
)

output_html = "/sessions/gifted-zealous-goldberg/mnt/outputs/glasser_atlas_3d.html"
print(f"Writing HTML to {output_html}...")
fig.write_html(
    output_html,
    include_plotlyjs=True,
    full_html=True,
    config={
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'displaylogo': False,
        'scrollZoom': True,
    }
)
file_size_mb = os.path.getsize(output_html) / (1024 * 1024)
print(f"HTML file size: {file_size_mb:.1f} MB")

# Preview
print("Writing preview PNG...")
fig.update_layout(width=1200, height=700)
fig.write_image("/sessions/gifted-zealous-goldberg/mnt/outputs/glasser_preview.png", scale=2)
print("Done!")
