# Third-party software and data

This repository does not redistribute dataset media, pretrained feature files,
or baseline implementations.

## Runtime dependencies

- PyTorch: BSD-style license.
- NumPy and SciPy: BSD licenses.
- h5py: BSD license.
- Matplotlib and Pillow are optional plotting dependencies under their own licenses.

The dependency packages are installed from their official distributions and are
not vendored here.

## Data and representations

MSCOCO, NUS-WIDE, and IAPR TC-12 remain subject to their original dataset terms.
The frozen CLIP representations are derived data and are not included. Users must
obtain the datasets and representations lawfully. CLIP is used only as a frozen
feature extractor in the reported primary experiments.

## Comparison methods

UIH, AMSH, GSPH, EDMH, LCDH, PIC-CMH, OCMFH, DOCH, OH-ELS, and SSOCH are cited
comparison methods. Their source code is not copied into this package. Bundled
CSV files contain only centrally evaluated numerical results and provenance
hashes. See `docs/BASELINES.md` for the access assumptions used in the paper.

