
# Deployment Verification

The deployed artifact is one fused student model, one checkpoint, and one image
view. Five teachers are used offline only and are absent from runtime.

## Locked artifact

- Submission ID: `879954`
- Immutable image: `docker.io/cjunxiao/foundus2026-final@sha256:4379314a848ae22acbdbcd12366fff255ae603ce1512c64ae448c0a77552a83e`
- Checkpoint SHA256: `73086754757a70be87319d636ab4b837aeeefbed88e2189add40484cfb7b8da1`
- Status: verified, deployed, and scored

## Compatible-GPU preflight

The published container root filesystem processed all 619
official-validation images on an NVIDIA RTX A6000 with the official PyTorch
2.3.1/CUDA 12.1 stack.

- Total wall-clock time: 35.36 seconds
- Throughput: 17.51 images/second
- Average wall-clock time: 57.12 ms/image
- Peak CUDA reserved memory: 1.63 GB

Anonymous Docker Hub access and the immutable digest were rechecked on
22 August 2026. The runtime is self-contained and writes the organizer-required
`/output/regression_predictions.json` file.
