# LTX-2 LoRA Guide

Source: https://docs.ltx.video/open-source-model/usage-guides/lo-ra.md

## Overview

LoRAs are lightweight model modifications requiring only 1-128MB of additional weights. They customize LTX-2's output for specific styles, effects, or visual characteristics without requiring full model fine-tuning.

## Key Capabilities

- Maintain visual consistency across generated videos through specific aesthetic guidelines
- Improve retention of character and object details during motion
- Customize movement interpretation for particular applications
- Implement structural control via depth maps, pose skeletons, or edge detection

## Official Camera Control LoRAs

Lightricks provides seven camera control LoRAs:
- Dolly movements (in, out, left, right)
- Jib movements (up, down)
- Static camera option

## ComfyUI Integration (Recommended)

Install ComfyUI-LTXVideo custom nodes, download LoRA files from Hugging Face, and place them in `ComfyUI/models/loras/`.

Alternative community loader:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dorpxam/ComfyUI-LTXVideoLoRA.git
```

## Strength Adjustment Guidelines

LoRA strength ranges from 0.0 to 1.0+:
- **0.9-1.1**: Subtle effects preserving base model characteristics
- **1.2-1.4**: Balanced approach for typical applications
- **1.5-1.6**: Strong style transfer effects

## Multi-LoRA Stacking

- Combined strength should remain below 2.0
- Effect LoRAs combine more successfully than control LoRAs

## Training Custom LoRAs

Custom LoRAs can be created using the LTX-2 Trainer repository with dedicated documentation available.
