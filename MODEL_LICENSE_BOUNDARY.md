
# Model and License Boundary

## Team-authored source

The repository-authored implementation is released under Apache License 2.0.
This license applies only to source written by the team and does not relicense
third-party software, pretrained weights, challenge data, or derived model
weights whose use is constrained by upstream terms.

## ConvNeXt initialization

The ConvNeXt-Small model card and timm software identify Apache-2.0 terms. The
checkpoint is used as an initialization resource and is not redistributed in
this compact materials package.

## USFM initialization and derived weights

USFM is distributed by its authors under CC BY-NC 4.0. `USFM_latest.pth` and
model weights derived from that initialization remain subject to the upstream
non-commercial and attribution conditions. The Apache-2.0 repository license
does not override those conditions.

## Challenge data and artifacts

Organizer-provided images, labels, metadata, and generated per-case targets are
not redistributed. The immutable challenge Docker is the executable reference;
access to the organizer data and the SHA-locked deployment checkpoint is still
required for local end-to-end reconstruction.

See `THIRD_PARTY_NOTICES.md` and the texts under `LICENSES/` before reuse.
