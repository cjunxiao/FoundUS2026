
# Third-Party Notices

## ConvNeXt-Small initialization

- Model: `timm/convnext_small.in12k_ft_in1k`
- Source: https://huggingface.co/timm/convnext_small.in12k_ft_in1k
- Pretraining: ImageNet-12K; fine-tuning: ImageNet-1K
- Model-card license: Apache-2.0
- Use: initialization of the supervised ConvNeXt heatmap pathway
- timm source code: Apache-2.0

## Ultrasound Foundation Model initialization

- Model: USFM, BEiT-B/16 backbone
- Source: https://github.com/openmedlab/USFM
- Weight: `USFM_latest.pth`
- Weight SHA256: `d5fdab3edd140e4ca61471bb4087f91cd7ff2ce270db71b9cab30feda881bd17`
- Repository license: CC BY-NC 4.0
- Public description: pretrained on approximately two million multi-organ,
  multi-center, multi-device ultrasound images
- Citation: Jiao et al., Medical Image Analysis 2024,
  https://doi.org/10.1016/j.media.2024.103202
- Use: initialization of the ultrasound-foundation heatmap pathway

## Challenge data

No external raw images or landmark annotations are added to the organizer data.
External datasets enter only through the two disclosed public pretrained
checkpoints above. Challenge data and generated teacher targets are not included
in this compact source package.

## Organizer-provided visual examples

Presentation overlays use organizer-provided challenge images only for method and result communication. Landmark overlays and labels were added by the team; no hidden-test image or label is included.

## Project code scope

The public method repository states the reuse terms for team-authored source. Third-party components remain governed by their original licenses.
