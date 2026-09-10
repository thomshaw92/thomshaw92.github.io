"""
core.py

Shared data structure (VertexLabels) and the base AtlasSource class.
AtlasSource's default fetch()/transform() implementation handles any
atlas shipped as a bundled FreeSurfer .annot file on MNE's fsaverage
(aparc, aparc.a2009s, PALS_B12_*, Yeo2011_*, HCPMMP1, HCPMMP1_combined,
oasis.chubs, etc.) - no subclassing needed for those.

Subclass AtlasSource (see sources.py) only when the fetch/transform
logic is genuinely different, e.g. volumetric nilearn atlases or
atlases downloaded from an arbitrary URL.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import nibabel as nib


@dataclass
class VertexLabels:
    """The shared intermediate format every transformer must produce."""
    labels: np.ndarray          # per-vertex integer label, shape (n_vertices,)
    names: list                 # label index -> region name (str)
    ctab: Optional[np.ndarray] = None  # Nx4/Nx5 RGBA(+id) color table, or None


def ensure_fsaverage_full():
    """Download full-resolution fsaverage (+ all MNE fsaverage
    parcellations, including HCPMMP1/HCPMMP1_combined) if not already
    cached. No-op if already present."""
    import mne
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    return fs_dir


def ensure_fsaverage6_surfaces(data_dir=None):
    """Download the real FreeSurfer fsaverage6 surface mesh via nilearn
    (not an MNE ico-decimation approximation)."""
    from nilearn import datasets as nil_datasets
    return nil_datasets.fetch_surf_fsaverage(mesh="fsaverage6", data_dir=data_dir)


class AtlasSource:
    """Base atlas source: default behaviour covers any atlas that is a
    bundled FreeSurfer .annot file living in fsaverage's label/ dir.

    Parameters
    ----------
    key : str
        Unique identifier, used for filenames / CLI selection.
    name, description, citation : str
        Display metadata.
    annot_filename : str
        e.g. "lh.aparc.annot". Required for the default fetch().
    resolution : str
        'fsaverage' | 'fsaverage6' | ... (informational / for downstream
        surface selection).
    license_note : str, optional
        Shown in the UI if the atlas carries usage/citation requirements.
    tags : list[str]
        Free-form tags, e.g. ['surface-native'], ['volumetric'].
    """

    def __init__(self, key, name, description, citation,
                 annot_filename=None, resolution="fsaverage6",
                 license_note=None, tags=None):
        self.key = key
        self.name = name
        self.description = description
        self.citation = citation
        self.annot_filename = annot_filename
        self.resolution = resolution
        self.license_note = license_note
        self.tags = tags or []

    def fetch(self) -> dict:
        """Return whatever transform() needs. Default: locate the bundled
        .annot file on fsaverage, downloading fsaverage if necessary."""
        fs_dir = ensure_fsaverage_full()
        annot_path = os.path.join(fs_dir, "label", self.annot_filename)
        if not os.path.exists(annot_path):
            raise FileNotFoundError(
                f"{self.annot_filename} not found in {fs_dir}/label - "
                f"this atlas may not be bundled with fetch_fsaverage()."
            )
        return {"annot_path": annot_path}

    def transform(self, fetch_result: dict) -> VertexLabels:
        """Default: read a FreeSurfer .annot into VertexLabels."""
        labels, ctab, names = nib.freesurfer.read_annot(fetch_result["annot_path"])
        names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in names]
        return VertexLabels(labels=labels, names=names, ctab=ctab)

    def build(self) -> VertexLabels:
        """Orchestration only - do not override this. Override fetch()
        and/or transform() instead."""
        fetch_result = self.fetch()
        return self.transform(fetch_result)

    def __repr__(self):
        return f"<{type(self).__name__} key={self.key!r} name={self.name!r}>"