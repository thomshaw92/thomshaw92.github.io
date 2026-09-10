"""
sources.py

AtlasSource subclasses for fetch/transform "shapes" that differ from
the default (bundled FreeSurfer .annot). Add a new subclass here only
when neither AtlasSource nor an existing subclass already covers the
new atlas's fetch/transform logic - most new atlases should just be a
new entry in registry.py using one of these classes with different
parameters.
"""

import os
import tempfile
from urllib.parse import urlparse

import numpy as np
import requests

from .core import AtlasSource, VertexLabels


class VolumetricAtlasSource(AtlasSource):
    """For atlases delivered as an MNI152 volume (most nilearn
    fetch_atlas_* functions), projected onto a surface mesh."""

    def __init__(self, *args, fetch_fn, fetch_kwargs=None,
                 surf_file=None, interpolation="nearest_most_frequent", radius=3.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_fn = fetch_fn
        self.fetch_kwargs = fetch_kwargs or {}
        self.surf_file = surf_file  # resolved lazily in fetch() if None
        self.interpolation = interpolation
        self.radius = radius
        self.name = self.name + ' - projected'

    def fetch(self) -> dict:
        atlas = self.fetch_fn(**self.fetch_kwargs)
        names = list(atlas.labels)
        surf_file = self.surf_file
        if surf_file is None:
            from .core import ensure_fsaverage_full
            fs_dir = ensure_fsaverage_full()
            surf_file = os.path.join(fs_dir, "surf", "lh.pial")
        return {"vol_img": atlas.maps, "names": names, "surf_file": surf_file}

    def transform(self, fetch_result: dict) -> VertexLabels:
        from nilearn import surface
        import nibabel as nib

        coords, faces = nib.freesurfer.read_geometry(fetch_result["surf_file"])
        texture = surface.vol_to_surf(
            fetch_result["vol_img"],
            (coords, faces),
            interpolation=self.interpolation,
            radius=self.radius,
        )
        labels = np.nan_to_num(texture, nan=0).astype(np.int32)
        return VertexLabels(labels=labels, names=fetch_result["names"], ctab=None)


class CustomURLAtlasSource(AtlasSource):
    """For a .annot file downloaded from an arbitrary URL. Reuses the
    base class's .annot transform() unchanged."""

    def __init__(self, *args, url, **kwargs):
        super().__init__(*args, **kwargs)
        self.url = url

    def fetch(self) -> dict:
        response = requests.get(self.url)
        response.raise_for_status()

        parsed_url = urlparse(self.url)
        filename = os.path.basename(parsed_url.path)
        if not filename.endswith(".annot"):
            filename += ".annot"

        tmp_dir = tempfile.mkdtemp()
        filepath = os.path.join(tmp_dir, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)

        return {"annot_path": filepath}
    # transform() inherited from AtlasSource - still a plain .annot file