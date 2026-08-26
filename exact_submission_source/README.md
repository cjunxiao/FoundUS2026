
# Exact Submitted Runtime Source

This directory preserves the source symbols and OCI labels used by final
CodaBench submission `879954`. A top-level `checkpoint_sha256` compatibility
field has been restored in `model_manifest.json`; it duplicates the immutable
nested checkpoint hash and does not change model behavior.

The immutable public Docker is the executable reference. The checkpoint is not
duplicated in this compact materials package, so local reconstruction requires
the SHA-locked checkpoint recorded in the manifest.
