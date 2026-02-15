# Workflows Medusa I2V

## 🎯 Deux versions disponibles

### V1 - Resize Nodes (ComfyUI_essentials)
**Fichier**: `v1-resize.json`

**Pipeline**:
1. Preprocess: Resize to 2M pixels → Align 64px → Half-res
2. Pass 1: Generation half-res (8 steps Euler)
3. VAE Decode Tiled
4. Post-resize: Scale to 720p (~0.92M pixels)

**Avantages**:
- ✅ Simple et rapide
- ✅ Contrôle précis de la résolution finale

**Inconvénients**:
- ❌ `ResizeImageMaskNode` pas maintenu depuis 10 mois
- ❌ Resize en pixel-space (après decode)
- ❌ Qualité limitée pour upscale

---

### V2 - Spatial Upscaler (LTX-2 officiel) ⭐ RECOMMANDÉ
**Fichier**: `v2-spatial-upscaler.json`

**Pipeline**:
1. Preprocess: Resize to 2M pixels → Align 64px → Half-res
2. **Pass 1**: Generation half-res (8 steps Euler)
3. **Spatial Upscaler**: LTX-2 x2 upscale dans l'espace latent (~1GB VRAM)
4. **Pass 2**: Refinement full-res (4 steps, denoise 0.4)
5. VAE Decode Tiled → Full-res video

**Avantages**:
- ✅ Spatial upscaler **officiel LTX-2**
- ✅ Upscale dans l'espace **latent** (meilleure qualité)
- ✅ Refinement v2v pour qualité optimale
- ✅ Léger: ~1GB VRAM pour l'upscaler

**Inconvénients**:
- ⚠️ Légèrement plus lent (4 steps supplémentaires)
- ⚠️ Nécessite téléchargement du modèle upscaler (~1GB)

---

## 📦 Modèles inclus

### Checkpoint & Text Encoder
- `ltx-2-19b-dev-fp8.safetensors` (checkpoint principal)
- `gemma_3_12B_it_fp8_scaled.safetensors` (text encoder, CPU)

### LoRAs
- `ltx-2-19b-distilled-lora-384.safetensors` (génération rapide 8 steps)
- `LTX-2-Image2Vid-Adapter.safetensors` (I2V adapter)
- `ltx-2-19b-lora-camera-control-dolly-in.safetensors` (effet camera dolly)

### Upscaler (V2 uniquement)
- `ltx-2-spatial-upscaler-x2-1.0.safetensors` (~1GB)

---

## 🚀 Utilisation

### Via RunPod API

**V1 - Resize**:
```json
{
  "input": {
    "workflow_name": "v1-resize.json",
    "inputs": {
      "10": { "image": "https://example.com/image.jpg" },
      "3": { "text": "Smooth dolly-in camera movement" }
    }
  }
}
```

**V2 - Spatial Upscaler** ⭐:
```json
{
  "input": {
    "workflow_name": "v2-spatial-upscaler.json",
    "inputs": {
      "10": { "image": "https://example.com/image.jpg" },
      "3": { "text": "Smooth dolly-in camera movement" }
    }
  }
}
```

### Scripts de test

```bash
# Test V1
./test-simple-inputs.sh

# Test V2 (recommandé)
./test-v2-spatial.sh
```

---

## 🎬 Effets Camera (LoRAs)

Swap le LoRA camera (node 7) pour différents effets :

- `ltx-2-19b-lora-camera-control-dolly-in.safetensors` (zoom avant) ✅
- `ltx-2-19b-lora-camera-control-dolly-out.safetensors` (zoom arrière)
- `ltx-2-19b-lora-camera-control-pan-left.safetensors` (panoramique gauche)
- `ltx-2-19b-lora-camera-control-pan-right.safetensors` (panoramique droite)
- etc.

**Note**: Télécharger les autres LoRAs camera depuis [Lightricks/LTX-2](https://huggingface.co/Lightricks) si besoin.

---

## ⚙️ Configuration

### Paramètres modifiables (nodes inputs)

**Node 10** - Input Image:
- `image`: URL ou nom fichier

**Node 3** - Positive Prompt:
- `text`: Description du mouvement/scène

**Node 4** - Negative Prompt:
- `text`: Ce qu'on veut éviter

**Node 33/87** - Seed:
- `noise_seed`: Random ou fixe pour reproductibilité

**Node 86** (V2 uniquement) - Refinement:
- `steps`: 4-8 steps (4 = rapide, 8 = qualité)
- `denoise`: 0.3-0.5 (force du refinement)

---

## 📊 Comparaison Performance

| Version | Pass 1 | Upscale | Pass 2 | Decode | Total (approx) |
|---------|--------|---------|--------|--------|----------------|
| **V1**  | ~30s   | -       | -      | ~20s   | **~50s**       |
| **V2**  | ~30s   | ~5s     | ~15s   | ~20s   | **~70s**       |

**Note**: Temps approximatifs sur GPU RunPod (RTX 4090 / A100)

---

## 🎯 Recommandation

**Utiliser V2 (Spatial Upscaler)** pour :
- ✅ Meilleure qualité finale
- ✅ Upscale natif LTX-2 (maintenu officiellement)
- ✅ Pipeline conforme aux best practices

**Utiliser V1 (Resize)** seulement si :
- ⚠️ Budget GPU très limité
- ⚠️ Vitesse critique (20s plus rapide)
