# Third-party software and data

This repository does not redistribute original dataset media or baseline
implementations. Derived feature files used in the reported experiments are
distributed separately through the download link documented in the repository
`README.md` and `docs/DATA.md`.

## Runtime dependencies

* PyTorch: BSD-style license.
* NumPy and SciPy: BSD licenses.
* h5py: BSD license.
* Matplotlib and Pillow are optional plotting dependencies under their own licenses.

The dependency packages are installed from their official distributions and are
not vendored here.

## Data and representations

MSCOCO, NUS-WIDE, and IAPR TC-12 remain subject to their original dataset terms.
Original dataset media are not redistributed by this project. Users are
responsible for complying with the applicable terms and licences of the
underlying datasets.

The derived feature files used in the reported experiments are made available
separately from the source repository. These include the frozen CLIP
representations used in the primary experiments and the MSCOCO VGG19+BOW
representation used in the robustness experiment. Download information, file
layout, feature schemas, and protocol-level hash verification details are
provided in `README.md` and `docs/DATA.md`.

CLIP is used only as a frozen feature extractor in the reported primary
experiments. The distributed feature files are derived representations and
should not be interpreted as a redistribution of the original dataset media.

## Comparison methods

UIH, AMSH, GSPH, EDMH, LCDH, PIC-CMH, OCMFH, DOCH, OH-ELS, and SSOCH are cited
comparison methods. Their source code is not copied into this package. Bundled
CSV files contain only centrally evaluated numerical results and provenance
hashes. See `docs/BASELINES.md` for the access assumptions used in the paper.
