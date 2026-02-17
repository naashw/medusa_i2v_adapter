# LTX-2 Image-to-Video Guide (ComfyUI)

Source: https://docs.ltx.video/open-source-model/usage-guides/image-to-video.md

## Overview

Generation de video a partir d'une image statique avec LTX-2 dans ComfyUI. Le workflow produit video et audio synchronises en une seule passe.

## Use Cases

- Animer des frames statiques avec effets de mouvement
- Preserver l'apparence des personnages et le cadrage exact
- Partir d'une composition visuelle connue

## Prerequisites

- ComfyUI installe avec les nodes LTX-2
- Image source prete
- Au moins 32GB VRAM (plus recommande)
- Workflow JSON Image-to-Video du repo LTX-2

## Model Configuration

Le node **LTXVCheckpointLoader** requiert :
- **Model selection** : distilled (plus rapide, optimise pour iteration) ou full checkpoint (meilleure qualite)
- VAE charge automatiquement avec le checkpoint
- Text encoder utilise Gemma CLIP pour le traitement des prompts

## Image Loading

Le node **LoadImage** accepte PNG, JPG et WebP. Les images sont automatiquement redimensionnees a la resolution cible. Choisir des images correspondant au ratio d'aspect desire.

## Resolution Parameters

Ratios d'aspect standard disponibles :
- **1920x1080** (16:9 paysage)
- **1080x1920** (9:16 portrait)
- **1280x720** (16:9 HD)
- **768x512** (3:2)
- **640x640** (1:1 carre)

> Les dimensions video doivent etre divisibles par 32. Les resolutions plus elevees necessitent plus de VRAM.

## Frame Configuration

- **Maximum frames** : 257 (~10 secondes a 25fps)
- **Range recommande** : 121-161 frames pour un equilibre qualite/memoire
- **Iterations rapides** : 65-97 frames

## Frame Rate Options

- **24 fps** : Apparence cinematique
- **25 fps** : Standard par defaut
- **30 fps** : Mouvement fluide pour l'action
- **48-60 fps** : Contenu dynamique ou rapide

> Le frame rate doit correspondre entre tous les nodes ou la vitesse de lecture sera incorrecte.

## Prompt Writing

Les prompts efficaces incluent :
- Description du mouvement (camera, changements de scene)
- Actions et comportement des personnages
- Type de plan et mouvement de camera
- Elements audio (dialogue, musique, sons ambiants)

Exemple : "Camera slowly zooms in while character turns to observe the sunset"

## Sampling Configuration

### Steps
- **Distilled model** : 8 steps (9 valeurs sigma manuelles)
- **Full model** : 20-50 steps

### CFG (Classifier-Free Guidance)
Range 2.0-5.0 :
- **2.0-3.0** : Plus creatif
- **4.0-5.0** : Meilleure adherence au prompt
- **Recommande** : 3.0-4.0

### Sampler types
- `euler` (rapide)
- `dpmpp_2m` (meilleure qualite)
- `Res-2-S` (full model)

### Seed
Fixe pour resultats reproductibles, random pour variations.

## Multi-Scale Generation (2 passes)

### Stage 1
Genere a demi-resolution (test rapide de composition et mouvement)

### Stage 2
Upscale a la resolution cible via le node **LTXVUpscale** avec magnification 2x.

Permet une experimentation efficace tout en maintenant la qualite.

## Audio-Video Processing

L'architecture unifiee :
1. Cree les latents audio et video independamment
2. Combine les latents pour traitement conjoint
3. Separe les streams, puis recombine pour un output coherent
4. Utilise le tiled decoding pour minimiser la demande memoire

### Audio Decoder
Traite les latents audio, produit du son synchronise (dialogue, musique, effets ambiants)

### Video Decoder
Utilise le tiled decoding pour gerer la VRAM efficacement tout en preservant la qualite

## Output Saving

Configuration du node **Save Video** :
- **Formats** : MP4 (defaut), MOV, WebM
- **Codecs** : H.264 (compatibilite) ou H.265 (fichiers plus petits)
- Audio automatiquement embarque depuis le decoder
- Utiliser des noms de fichiers descriptifs

## Advanced: Full Model Workflow

Le full model offre une qualite amelioree avec des temps de traitement plus longs.

Differences :
- Utilise le checkpoint LTX-2 complet et un VAE specialise
- Stage 1 : 15-20 steps (jusqu'a 40 pour experimentation)
- Utilise **LTXV Scheduler** au lieu de sigmas manuels
- Applique un **distilled LoRA** en Stage 2 (strength recommandee : 0.6)

## LoRA Application

Les nodes **LoRALoader** permettent :
- **Style LoRAs** : Appliquer des esthetiques visuelles
- **Motion LoRAs** : Ameliorer les types de mouvement
- **Character LoRAs** : Maintenir une apparence coherente
