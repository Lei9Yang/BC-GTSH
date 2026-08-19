# Comparison tracks

The public package does not vendor third-party implementations. This avoids
changing their license terms and prevents local author-repository paths from
becoming part of the release.

## Strict source-disjoint online track

UIH is evaluated with the same fixed queries and append-only gallery as BC-GTSH.
Its data adapter receives the two disjoint modality streams. The bundled values
are a unified-protocol re-evaluation, not the values from UIH's native resampled
MIRFlickr protocol.

## Offline unpaired references

AMSH, GSPH, and EDMH receive the union of all ten training stages and produce a
single final-gallery result. They do not produce synthetic online curves. GSPH
uses the released solver's 2,000-sample training limit. These results quantify
the benefit of complete final-data access and are not ranked as online methods.

## Paired-source Oracle track

LCDH, PIC-CMH, OCMFH, DOCH, OH-ELS, and SSOCH may access paired source IDs.
BC-GTSH-PairBlind uses the same source pool, independently permutes image and
text order, and removes pair indices from the method input. It measures behavior
under a paired source pool without claiming strict source-disjoint training.

External implementations should export image/text binary codes, gallery labels,
query labels, source IDs, bit length, seed, and protocol SHA-256. Central mAP
evaluation then uses label-intersection relevance.

