# Data format and frozen protocols

## Feature schema

The three formal datasets use MATLAB v7.3/HDF5 containers:

| Field | Meaning | Stored shape |
|---|---|---|
| `Images` | frozen image representation | feature dimension x samples |
| `Texts` | frozen text representation | feature dimension x samples |
| `Labels` | multi-hot labels | classes x samples |
| `Idx` | unique source IDs | 1 x samples |

The primary experiments use 512-dimensional CLIP representations. Labels are
used during training to construct the privileged teacher graph and during
offline evaluation; they are not an inference input.

Expected dataset hashes are stored in each protocol `manifest.json`. Run
`bc-gtsh data audit` rather than relying on a filename alone.

For compatibility with the original experiment workspace, the CLI also accepts
`data/nuswide-TC81/NUSwide_CLIP_image_CLIP_text.mat` when the preferred
`data/nuswide81/` path is absent.

## Protocol layout

`protocols/<dataset>/strict/{validation,test}/seed-6513/` contains a sanitized
manifest and deterministic `indices.npz`. The model seed does not alter this
strict split. PairBlind has a separately frozen permutation for every model seed
under `protocols/<dataset>/pairblind/test/seed-<seed>/`.

The verifier checks:

- validation and test query sets are disjoint;
- queries never enter the training stream;
- strict image and text training source sets have zero overlap;
- a training source occurs in exactly one stage;
- ten stages cover the complete training pool;
- the protocol indices and feature file match their SHA-256 records.

The protocol manifest stores a release-relative dataset path. This field is
informational; the hash is authoritative.

## Alternate representations

The bundled evidence includes the paper's MSCOCO VGG19+BOW robustness result,
but the alternate feature file and extractor repository are not redistributed.
Users may provide an HDF5 file with the same four fields and run the public
method after defining an equivalent dataset specification.
