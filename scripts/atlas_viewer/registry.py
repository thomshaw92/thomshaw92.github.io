"""
registry.py

Pure data: maps atlas key -> AtlasSource instance. No fetch/transform
logic lives here - only instantiations. Adding a bundled-annot atlas
is a one-line entry using the base AtlasSource; adding a new *type* of
atlas means adding a subclass in sources.py first, then one entry here.
"""

import glob
import os
from pathlib import Path
import urllib.request

from .core import AtlasSource, ensure_fsaverage_full
from .sources import VolumetricAtlasSource, CustomURLAtlasSource

REMOTE_ATLAS_URL = (
    "https://raw.githubusercontent.com/thomshaw92/"
    "FingertipMappingMND/feature/bodymap-preprocessing/"
    "bodymap_preprocessing/atlas.py"
)


def _download_remote_atlas_module():
    """Download a missing top-level atlas.py module from one of our
    other repos that carries fetchers for additional atlases.

    This keeps the registry loader self-healing: the import hook can
    be satisfied from the user workspace, or a tiny network fallback can
    fetch the expected atlas.py file.
    """
    repo_root = Path(__file__).resolve().parent
    local_atlas = repo_root / "atlas.py"
    if local_atlas.exists():
        return local_atlas

    try:
        with urllib.request.urlopen(REMOTE_ATLAS_URL, timeout=30) as response:
            body = response.read()
        local_atlas.write_bytes(body)
    except Exception:
        return None
    return local_atlas


# Bundled FreeSurfer .annot atlases (ship with mne.datasets.fetch_fsaverage)
ATLAS_REGISTRY = {
    "aparc": AtlasSource(
        key="aparc", name="Desikan-Killiany Atlas",
        description="Desikan-Killiany Atlas (2006)",
        citation="Desikan et al., 2006",
        annot_filename="lh.aparc.annot",
    ),
    "aparc.a2009s": AtlasSource(
        key="aparc.a2009s", name="Destrieux Atlas",
        description="Destrieux Atlas (2009)",
        citation="Destrieux et al., 2010",
        annot_filename="lh.aparc.a2009s.annot",
    ),
    "aparc.a2005s": AtlasSource(
        key="aparc.a2005s", name="Destrieux 2005 Atlas",
        description="Destrieux Atlas (2005)",
        citation="Destrieux et al., 2010",
        annot_filename="lh.aparc.a2005s.annot",
    ),
    "HCPMMP1": AtlasSource(
        key="HCPMMP1", name="Glasser Atlas (HCP-MMP1.0)",
        description="HCP-MMP1.0 Parcellation",
        citation="Glasser et al., 2016",
        annot_filename="lh.HCPMMP1.annot",
        license_note=(
            "Acknowledge the use of WU-Minn HCP data and data derived "
            "from WU-Minn HCP data when publicly presenting results."
        ),
    ),
    "HCPMMP1_combined": AtlasSource(
        key="HCPMMP1_combined", name="Glasser Atlas (Combined, 23 regions)",
        description="HCP-MMP1.0 Parcellation, combined/reduced set",
        citation="Glasser et al., 2016",
        annot_filename="lh.HCPMMP1_combined.annot",
        license_note=(
            "Acknowledge the use of WU-Minn HCP data and data derived "
            "from WU-Minn HCP data when publicly presenting results."
        ),
    ),
    "Yeo2011_7Networks": AtlasSource(
        key="Yeo2011_7Networks", name="Yeo 7 Networks",
        description="Yeo 7 Resting-State Networks",
        citation="Yeo et al., 2011",
        annot_filename="lh.Yeo2011_7Networks_N1000.annot",
    ),
    "Yeo2011_17Networks": AtlasSource(
        key="Yeo2011_17Networks", name="Yeo 17 Networks",
        description="Yeo 17 Resting-State Networks",
        citation="Yeo et al., 2011",
        annot_filename="lh.Yeo2011_17Networks_N1000.annot",
    ),
    "PALS_B12_Lobes": AtlasSource(
        key="PALS_B12_Lobes", name="PALS Lobe Atlas",
        description="PALS Lobe Parcellation",
        citation="Van Essen, 2005",
        annot_filename="lh.PALS_B12_Lobes.annot",
    ),
    "PALS_B12_Brodmann": AtlasSource(
        key="PALS_B12_Brodmann", name="PALS Brodmann Atlas",
        description="PALS Brodmann Areas",
        citation="Van Essen, 2005",
        annot_filename="lh.PALS_B12_Brodmann.annot",
    ),
    "oasis.chubs": AtlasSource(
        key="oasis.chubs", name="OASIS CHUBS Atlas",
        description="OASIS CHUBS Parcellation",
        citation="OASIS",
        annot_filename="lh.oasis.chubs.annot",
    ),
}

#Additional atlases from nilearn datasets, if available.
from nilearn import datasets as nil_datasets
try:
    import atlas
except Exception:
    got_atlas = _download_remote_atlas_module()
    if got_atlas is not None:
        import atlas

if hasattr(nil_datasets, "fetch_atlas_harvard_oxford"):
    ATLAS_REGISTRY["harvard_oxford_cort"] = VolumetricAtlasSource(
        key="harvard_oxford_cort", name="Harvard-Oxford Cortical Atlas",
        description="Probabilistic cortical atlas, max-probability thresholded",
        citation="Makris et al., 2006",
        fetch_fn=nil_datasets.fetch_atlas_harvard_oxford,
        fetch_kwargs={"atlas_name": "cort-maxprob-thr25-2mm"},
        tags=["volumetric"],
        radius=5.0,
    )

if hasattr(nil_datasets, "fetch_atlas_schaefer_2018"):
    ATLAS_REGISTRY["schaefer_2018"] = VolumetricAtlasSource(
        key="schaefer_2018", name="Schaefer 2018 Atlas",
        description="Deterministic Schaefer 2018 parcellation, 400 regions, 7 networks",
        citation="Alexander Schaefer, Ru Kong, Evan M Gordon, Timothy O Laumann, Xi-Nian Zuo, Avram J Holmes, Simon B Eickhoff, and B T Thomas Yeo. Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. Cerebral Cortex, 28(9):3095-3114, 07 2017. doi:10.1093/cercor/bhx179.",
        fetch_fn=nil_datasets.fetch_atlas_schaefer_2018,
        fetch_kwargs={},
        tags=["volumetric"],
        radius=5.0,
    )

if hasattr(nil_datasets, "fetch_atlas_hmat_2006"):
    ATLAS_REGISTRY["hmat_2006"] = VolumetricAtlasSource(
        key="hmat_2006", name="HMAT 2006 Atlas",
        description="Human Motor Area Template atlas with motor areas only in MNI152NLin6Asym space",
        citation="Mayka, M.A., Corcos, D.M., Leurgans, S.E., Vaillancourt, D.E. 2006. Three-dimensional locations and boundaries of motor and premotor cortices as defined by functional brain imaging: a meta-analysis. Neuroimage. 31(4):1453-1474. doi: 10.1016/j.neuroimage.2006.02.004.",
        fetch_fn=nil_datasets.fetch_atlas_hmat_2006,
        fetch_kwargs={},
        tags=["volumetric"],
        radius=5.0,
    )


def discover_unregistered_atlases():
    """Find bundled .annot files on fsaverage that aren't in
    ATLAS_REGISTRY yet, and wrap them in a generic AtlasSource. Useful
    as a fallback so nothing is silently missed, without requiring
    every atlas to be hand-registered."""
    fs_dir = ensure_fsaverage_full()
    label_dir = os.path.join(fs_dir, "label")
    registered_filenames = {
        src.annot_filename for src in ATLAS_REGISTRY.values()
        if getattr(src, "annot_filename", None)
    }

    extra = {}
    for annot_path in sorted(glob.glob(os.path.join(label_dir, "lh.*.annot"))):
        filename = os.path.basename(annot_path)
        if filename in registered_filenames:
            continue
        key = filename.replace("lh.", "").replace(".annot", "")
        if key in ATLAS_REGISTRY or key in extra:
            continue
        extra[key] = AtlasSource(
            key=key,
            name=key.replace("_", " ").title(),
            description=f"{key} atlas",
            citation="Unknown",
            annot_filename=filename,
        )
    return extra


def full_registry(include_discovered=True):
    """ATLAS_REGISTRY plus any unregistered bundled atlases found on
    disk, unless include_discovered=False."""
    registry = dict(ATLAS_REGISTRY)
    if include_discovered:
        registry.update(discover_unregistered_atlases())
    return registry