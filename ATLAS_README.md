# Interactive Brain Atlas System

This repository contains an interactive web-based brain atlas viewer that supports multiple cortical parcellation schemes available through MNE-Python and FreeSurfer.

## Features

- **Multiple Atlas Support**: Automatically discovers and builds all available atlases from MNE-Python
- **Interactive 3D Visualization**: Web-based Plotly visualization with hover tooltips
- **Surface Views**: Toggle between folded (pial) and inflated cortical surfaces
- **Custom Atlas Support**: Download and process custom atlas annotation files
- **Dynamic Loading**: Atlases load on-demand for fast initial page loads

## Available Atlases

The system currently includes these atlases (automatically discovered from MNE):

- **Glasser Atlas (HCP-MMP1.0)**: 180-region parcellation based on multi-modal MRI
- **Destrieux Atlases**: Anatomically-based parcellations (2005 and 2009 versions)
- **Desikan-Killiany Atlas**: 36-region anatomical parcellation
- **Yeo Networks**: Functional connectivity networks (7 and 17 network versions)
- **PALS Atlases**: Various PALS parcellations (Brodmann, Lobes, etc.)
- **OASIS CHUBS**: OASIS CHUBS parcellation
- **And more**: All atlases in your MNE installation

## Usage

### Building Atlases

Use the `scripts/build_atlas.py` script to manage atlases:

```bash
cd scripts

# List all available atlases
python build_atlas.py list

# Build all available atlases
python build_atlas.py build-all

# Build a specific atlas
python build_atlas.py build HCPMMP1

# Download and build a custom atlas
python build_atlas.py custom <url> [name] [description] [citation]
```

### Custom Atlases

To add a custom atlas:

1. Find or create a FreeSurfer `.annot` file for the left hemisphere
2. Host it somewhere accessible via URL
3. Run: `python build_atlas.py custom <url> "My Custom Atlas" "Description" "Citation"`

The system will:
- Download the annotation file
- Process it with the fsaverage surface
- Generate the interactive data
- Add it to the registry for web display

### Web Interface

- Open `brain_atlas.html` in a web browser
- Select any available atlas from the dropdown
- Toggle between folded and inflated views
- Hover over regions to see names
- Click and drag to rotate, scroll to zoom

## Technical Details

### Data Processing

- Uses full-resolution fsaverage surfaces (163k vertices)
- Computes per-face labels via majority voting
- Generates boundary lines between regions
- Creates optimized JSON data for web display

### File Structure

```
atlas_data_*.json      # Individual atlas data files
atlas_registry.json     # Registry of available atlases
brain_atlas.html        # Main web interface
scripts/build_atlas.py  # Atlas building and management script
```

### Dependencies

- Python: numpy, nibabel, plotly, scipy, mne, requests
- Web: Modern browser with JavaScript enabled

## Adding New Atlases

### From MNE/FreeSurfer

New atlases added to MNE will be automatically discovered. Just run:

```bash
python build_atlas.py build-all
```

### Custom Atlases

For custom atlases, you need:

1. A FreeSurfer `.annot` file for the left hemisphere
2. Proper alignment to fsaverage space
3. A URL where the file can be downloaded

Example:
```bash
python build_atlas.py custom \
  "https://example.com/my_atlas.annot" \
  "My Custom Atlas" \
  "Custom parcellation based on my research" \
  "Smith et al., 2024"
```

## References

- Glasser et al. (2016). A multi-modal parcellation of human cerebral cortex. Nature
- Destrieux et al. (2010). Automatic parcellation of human cortical gyri and sulci using standard anatomical nomenclature. NeuroImage
- Desikan et al. (2006). An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. NeuroImage
- Yeo et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. Journal of Neurophysiology
- Van Essen (2005). A Population-Average, Landmark- and Surface-based (PALS) atlas of human cerebral cortex. NeuroImage

## License

This atlas viewer is provided for informational, educational, and research purposes only. Not intended for clinical use.</content>
<parameter name="filePath">/home/uqmtoth/repos/thomshaw92.github.io/ATLAS_README.md