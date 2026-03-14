# Audit Pipeline Medusa I2V — Rapport d'Optimisation

**Date** : 2026-03-09
**Scope** : Pipeline LTX-2.3 22B Distilled, RunPod Serverless H100 80GB
**Methode** : Lecture integrale du codebase + verification documentation officielle + recherche web

---

## 1. Audit du stack actuel

| Composant | Version installee | Derniere version | A jour ? |
|-----------|-------------------|------------------|----------|
| ltx-core | Commit `9e8a28e` (2026-03-05) | Commit `9e8a28e` (latest sync PR #136) | Oui |
| ltx-pipelines | Commit `9e8a28e` | Idem | Oui |
| SageAttention | `main` HEAD (clone `--depth 1`) | 2.2.0 (release + fixes SM90) | Indetermine — depend de la date de build Docker |
| PyTorch | `>=2.7.1,<3` (cu128) | **2.10.0** (21 jan 2026, cu128 dispo) | **Non** — 4 versions majeures de retard |
| Triton | Installe via PyTorch (~3.2) | **3.6.0** (20 jan 2026, via torch 2.10) | **Non** — lie a PyTorch |
| flash-attn | Non installe | 2.8.3 (PyPI) / FA3 beta / **FA4 4.0.0b4** (5 mars 2026) | N/A |
| transformers | `>=4.52,<5.0` | **5.3.0** (4 mars 2026) — v5 BREAKING | Pin correct (`<5.0` nous protege) |
| huggingface-hub | `>=0.28` | **1.6.0** (6 mars 2026) | Non — mais compatible |
| runpod | `>=1.7,<2.0` | **1.8.1** (19 nov 2025) | Non — 1.8.0 a lazy-load boto3 |
| boto3 | `>=1.34,<2.0` | 1.42.63 (6 mars 2026) | Compatible |
| CUDA | 12.8.1 (cudnn-devel/runtime) | 12.8.1 | Oui |

**Notes critiques** :
- Le Dockerfile fait `git clone --depth 1 https://github.com/thu-ml/SageAttention` sans pin de version/tag. La version installee depend du moment du build. SageAttention 2.2.0 a eu un bug SM90 signale (issue #320) qui a ete corrige sur `main`. Si l'image Docker a ete buildee entre la release 2.2.0 et le fix, le kernel SM90 peut produire des artefacts.
- **PyTorch 2.10.0** est disponible depuis janvier 2026 avec des ameliorations potentiellement utiles (combo-kernels horizontal fusion, varlen_attn). Le pin `>=2.7.1,<3` l'autorise mais l'image Docker utilise probablement 2.7.1 ou 2.7.x selon la date du build.
- **transformers v5** (sortie 26 jan 2026) casse `Gemma3TextConfig` — notre pin `<5.0` est correct et critique. Derniere v4.x safe : **4.57.6** (16 jan 2026).
- **runpod 1.8.0** ajoute le lazy-load de boto3/fastapi/pydantic → cold start plus rapide. Upgrade gratuit.
- **Flash Attention 4** est sorti en beta (4.0.0b4, 5 mars 2026) — ecrit en CuTeDSL, supporte Hopper + Blackwell, integre torch.compile + FlexAttention. A surveiller.
- **SageAttention 3** existe (NeurIPS 2025 Spotlight) mais cible **Blackwell uniquement** (FP4 Tensor Cores SM120). Pas utilisable sur H100.

---

## 2. Etat actuel du pipeline

### Architecture
- **Checkpoint** : LTX-2.3 22B Distilled BF16 (`ltx-2.3-22b-distilled.safetensors`, ~46GB)
- **Quantization** : `QuantizationPolicy.fp8_cast()` — stockage FP8 (~19-20GB VRAM), compute BF16
- **Text encoder** : Gemma 3 12B IT (BF16, HuggingFace format)
- **Attention** : SageAttention2++ via patch runtime sur modules `Attention` du transformer, `torch.compiler.disable(sageattn)`, fallback SDPA si mask present
- **Compilation** : `torch.compile(mode="default")` quand SageAttention actif, `mode="reduce-overhead"` sinon
- **VAE** : Video decoder compile (`reduce-overhead`), video encoder non compile, tiling desactive (`None`)
- **Sampler** : `EulerDiffusionStep` (premier ordre), 8 steps stage 1, 3 steps stage 2

### Modes de generation
- **720p** : 1-stage, ~0.92M px, align 32px, 8 steps distilled
- **1080p** : 2-stage (540p → spatial upscale x2 latent → refine 1080p), 8+3 steps
- **Batch** : N images en un seul forward transformer, VAE decode sequentiel

### Optimisations presentes
- Eager init avant `runpod.serverless.start()`
- Embeddings pre-caches (7 presets + 1 negative)
- Transformer cache pre-fusionne sur volume (invalidation par hash)
- Post-processing async (MP4 encode + S3 upload en thread pool)
- Pipeline overlapping (prefetch images du batch suivant)
- Dedup cache par hash input
- `CFG=1.0, STG=0.0` — un seul forward par step
- Audio desactive
- Warmup embeddings en process isole (cleanup VRAM complet)
- tcmalloc

---

## 3. Bug connu : SageAttention + torch.compile

### Statut : DEJA CORRIGE (commit 7aedf21)

### Root cause
`torch.compile(mode="reduce-overhead")` tente de tracer `sageattn` via Dynamo. `sageattn` utilise des kernels Triton/pybind11 incompatibles avec FakeTensors (shape inference sans donnees reelles). Erreur : `"Cannot access data pointer of FakeTensor"`.

### Workarounds testes et fixes
1. `torch._dynamo.allow_in_graph(sageattn)` → Dynamo inclut sageattn dans le FX graph, l'execute avec FakeTensors → meme crash
2. `torch.compiler.disable(sageattn)` → cree des graph breaks. Avec `mode="reduce-overhead"`, les sous-graphes entre les breaks sont trop petits pour capturer des CUDA graphs utiles → output noise/bruit
3. **Fix actuel** (commit 7aedf21) : `torch.compiler.disable(sageattn)` + `mode="default"` quand SageAttention actif. `mode="reduce-overhead"` seulement si `SAGE_ATTENTION=0` (SDPA)

### Solution alternative : SageAttention 2.2.0
SageAttention 2.2.0 ajoute le support natif de `torch.compile`. Le `sageattn` peut etre trace par Dynamo sans graph breaks. **Source** : recherche web sur les releases SageAttention 2.2.0, confirmation dans les notes de release du fork woct0rdho/SageAttention.

**Impact** : Si SageAttention 2.2.0 est installe avec le fix SM90, on peut revenir a `mode="reduce-overhead"` avec SageAttention sans `compiler.disable`, eliminant les graph breaks et beneficiant des CUDA graphs.

**Caveat** : Le support `torch.compile` dans SageAttention 2.2.0 a ete signale comme cassant le backend SM90 (issue #320). Le fix est sur `main` mais il faut verifier que le build Docker utilise bien le code post-fix.

---

## 4. Opportunites d'optimisation

### 4.1. SageAttention 2.2.0 + torch.compile(reduce-overhead)

| | |
|---|---|
| **Description** | Mettre a jour SageAttention vers 2.2.0 (post-fix SM90) et supprimer `torch.compiler.disable`. Passer en `mode="reduce-overhead"` pour capturer des CUDA graphs complets |
| **Source** | [SageAttention releases](https://github.com/thu-ml/SageAttention/releases), [Issue #320 (fixed)](https://github.com/thu-ml/SageAttention/issues/320) |
| **Gain estime** | `mode="reduce-overhead"` vs `mode="default"` : elimination du overhead de lancement de kernels CUDA a chaque step. Gain non quantifie officiellement pour ce modele specifique, mais `reduce-overhead` est generalement 10-30% plus rapide que `default` pour les modeles avec peu de steps |
| **Effort** | Bas — pin un tag/commit SageAttention dans le Dockerfile, supprimer `compiler.disable`, changer la logique `compile_mode` |
| **Priorite** | **HAUTE** |
| **Risque** | Regression SM90 si le pin est mal fait. Tester sur H100 avant deploy |

### 4.2. torch.compile mode="max-autotune-no-cudagraphs"

| | |
|---|---|
| **Description** | Si SageAttention 2.2.0 + reduce-overhead pose encore des problemes, `mode="max-autotune-no-cudagraphs"` offre l'autotuning Triton des matmuls sans CUDA graphs |
| **Source** | [PyTorch docs torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) |
| **Gain estime** | L'autotuning selectionne les meilleurs kernels Triton/templates pour chaque matmul. Gain par rapport a `default` non quantifie pour ce modele |
| **Effort** | Bas — changer `mode="default"` en `mode="max-autotune-no-cudagraphs"` |
| **Priorite** | **MOYENNE** (fallback si 4.1 ne fonctionne pas) |

### 4.3. Sampler Res2s (second ordre) pour le pipeline 2-stage

| | |
|---|---|
| **Description** | ltx-core fournit `Res2sDiffusionStep`, un sampler de second ordre avec injection de bruit SDE. Utilise officiellement par `TI2VidTwoStagesHQPipeline` (anciennement `TI2VidTwoStagesRes2sPipeline`) |
| **Source** | [ltx-core/components/diffusion_steps.py](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/components/diffusion_steps.py), PR #136 (rename HQ) |
| **Gain estime** | Meilleure qualite a nombre de steps egal, ou qualite equivalente avec moins de steps. Le pipeline HQ officiel utilise 15 steps (vs 8+3 actuel). Gain non quantifie en termes de speed vs quality tradeoff |
| **Effort** | Moyen — remplacer `EulerDiffusionStep()` par `Res2sDiffusionStep()`, ajuster les sigmas, tester la qualite |
| **Priorite** | **MOYENNE** (qualite, pas speed) |

### 4.4. FP8 Scaled Matrix Multiplication (TensorRT-LLM path)

| | |
|---|---|
| **Description** | `QuantizationPolicy.fp8_scaled_mm()` utilise les FP8 matmuls natifs du Hopper (SM90) via TensorRT-LLM. Compute directement en FP8 au lieu de upcast BF16 |
| **Source** | [ltx-core/quantization](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core), [ltx-pipelines README](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md) |
| **Gain estime** | Les FP8 matmuls natifs sur H100 sont theoriquement ~2x plus rapides que BF16. Gain reel sur DiT non quantifie officiellement. VRAM similaire (~19-20GB) |
| **Effort** | **Haut** — necessite l'installation de `tensorrt_llm` dans l'image Docker (grosse dependance), potentielle calibration amax, tests de qualite approfondis |
| **Priorite** | **BASSE** (gros effort, gain incertain sans benchmarks publies) |

### 4.5. Pin SageAttention a un commit/tag specifique

| | |
|---|---|
| **Description** | Le Dockerfile fait `git clone --depth 1` sans pin. La version installee depend du moment du build, ce qui rend les builds non-reproductibles et expose au bug SM90 (#320) |
| **Source** | [Dockerfile:65](Dockerfile), [Issue #320](https://github.com/thu-ml/SageAttention/issues/320) |
| **Gain estime** | Aucun gain de performance, mais stabilite et reproductibilite des builds |
| **Effort** | Bas — ajouter `git checkout <commit>` apres le clone |
| **Priorite** | **HAUTE** (stabilite) |

### 4.6. API update : image_conditionings_by_replacing_latent → combined_image_conditionings

| | |
|---|---|
| **Description** | PR #136 (commit 9e8a28e) renomme `image_conditionings_by_replacing_latent` en `combined_image_conditionings` dans ltx-pipelines. Le pipeline actuel utilise encore l'ancien nom. Tant que le code est pin au commit 9e8a28e, ca fonctionne, mais une mise a jour du pin casserait |
| **Source** | PR #136 files changed — `utils/__init__.py` |
| **Gain estime** | Aucun. Preparation pour updates futures |
| **Effort** | Bas — rename dans pipeline.py |
| **Priorite** | **BASSE** (proactif) |

### 4.7. Upgrade PyTorch 2.7.x → 2.10.0

| | |
|---|---|
| **Description** | PyTorch 2.10.0 (21 jan 2026) apporte : combo-kernels horizontal fusion (fusionne ops GPU independantes en un kernel), `varlen_attn()` (attention ragged/packed compilable), Triton 3.6.0 (TMEM improvements Hopper). Wheel cu128 disponible |
| **Source** | [PyTorch 2.10 Release Blog](https://pytorch.org/blog/pytorch-2-10-release-blog/), [Triton 3.6.0 Releases](https://github.com/triton-lang/triton/releases) |
| **Gain estime** | Combo-kernels pourrait reduire le kernel launch overhead dans le pipeline de denoising. Triton 3.6.0 apporte des ameliorations TMEM pour Hopper. Gain non quantifie pour ce modele specifique |
| **Effort** | **Moyen** — changer le pin PyTorch dans le Dockerfile (`torch>=2.10,<3`), rebuild complet, tester la compatibilite SageAttention + ltx-core + torch.compile |
| **Priorite** | **MOYENNE** (4 versions majeures de retard, mais risque de regression) |
| **Risque** | SageAttention compile pour SM90 doit etre re-teste avec le nouveau Triton. ltx-core doit etre valide |

### 4.8. Upgrade runpod 1.7.x → 1.8.1

| | |
|---|---|
| **Description** | runpod 1.8.0 ajoute le lazy-load de boto3, fastapi, pydantic — les imports lourds ne sont charges que quand necessaires. Reduit le cold start du worker serverless |
| **Source** | [RunPod Python Releases](https://github.com/runpod/runpod-python/releases) |
| **Gain estime** | Cold start plus rapide (boto3 importe a la demande au lieu du startup). Gain non quantifie |
| **Effort** | **Bas** — changer `runpod>=1.8,<2.0` dans requirements.txt |
| **Priorite** | **HAUTE** (effort minimal, risque nul) |

### 4.9. Tiled VAE decoding pour le mode 1080p

| | |
|---|---|
| **Description** | ltx-core supporte `TilingConfig` avec `SpatialTilingConfig` et `TemporalTilingConfig` dans `decode_video`. Actuellement, le pipeline passe `None` (pas de tiling). Pour le mode 1080p, le VAE decode un latent ~2M px sans tiling |
| **Source** | [ltx-core/model/video_vae](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core/src/ltx_core/model/video_vae) |
| **Gain estime** | Reduction de la VRAM pic pendant le decode. Sur H100 80GB, pas critique (marge ~40-45GB). Pourrait etre utile si batch_size augmente |
| **Effort** | Bas — passer un `TilingConfig.default()` au `vae_decode_video`. Risque : ghosting temporal documente (ComfyUI issue #11767) |
| **Priorite** | **BASSE** (pas de contrainte VRAM actuelle) |

---

## 5. Analyse des alternatives

### 5.1. Attention kernel : SageAttention2++ vs Flash Attention 3

| Critere | SageAttention2++ | Flash Attention 3 |
|---------|-----------------|-------------------|
| H100 SM90 | Oui (INT8-QK/FP8-PV) | Oui (Hopper-only) |
| BF16 | Oui | Oui |
| torch.compile | Oui (v2.2.0) | Non verifie |
| Installation | `pip install` from source | Separate : `cd hopper && python setup.py install` (beta) |
| PyPI | Non (build from source) | Non pour FA3 (FA2 = flash-attn 2.8.3 sur PyPI) |
| Benchmarks H100 DiT | Non publies | Non publies pour DiT |
| Integration ltx-core | Via patch `attention_function` | Necessiterait patch custom |

**Verdict** : **Garder SageAttention2++**. Flash Attention 3 est en beta, pas sur PyPI, installation complexe, et aucun benchmark publie sur DiT. SageAttention est deja integre et fonctionne. Le seul avantage potentiel de FA3 serait la compatibilite torch.compile native, mais SageAttention 2.2.0 l'a aussi desormais.

### 5.2. Inference : torch.compile vs TensorRT

| Critere | torch.compile | TensorRT |
|---------|--------------|----------|
| Support ltx-core | Oui (natif PyTorch) | Via `fp8_scaled_mm` + tensorrt_llm |
| Compilation | JIT, premiere execution lente | AOT, necessite export |
| Flexibilite | Shapes dynamiques via guards | Shapes statiques ou profiles |
| Effort | Deja en place | Haut — tensorrt_llm ~10GB+ dans l'image |
| Serverless cold start | Premier job lent (compilation) | Froid mais rapide apres |

**Verdict** : **Garder torch.compile**. TensorRT via tensorrt_llm ajouterait une enorme complexite et taille d'image Docker pour un gain non quantifie. Le cache transformer pre-fusionne reduit deja le cold start.

### 5.3. FP8 : fp8_cast vs fp8_scaled_mm

| Critere | fp8_cast (actuel) | fp8_scaled_mm |
|---------|-------------------|---------------|
| Stockage | FP8 (~19-20GB) | FP8 (~19-20GB) |
| Compute | BF16 (upcast a chaque forward) | FP8 natif (Hopper SM90) |
| Dependance | Aucune | tensorrt_llm |
| Qualite | BF16 compute = pas de perte | FP8 compute = legere perte potentielle |
| Speed theorique | 1x | ~1.5-2x (FP8 matmuls H100) |
| Benchmarks publies | Non | Non |

**Verdict** : **Garder fp8_cast**. Le passage a fp8_scaled_mm necessite tensorrt_llm (grosse dep), aucun benchmark publie sur ce modele, et risque de regression qualite. A reconsiderer si Lightricks publie des benchmarks officiels.

### 5.4. VAE decoder : persistent sans tiling vs tiled

**Verdict** : **Garder sans tiling**. H100 80GB a suffisamment de VRAM. Le tiling ajoute du overhead de compute (overlap regions) et un risque de ghosting temporal documente. A reconsiderer seulement si le batch_size 1080p augmente significativement.

### 5.5. Sampler : Euler vs Res2s

| Critere | Euler (actuel) | Res2s |
|---------|---------------|-------|
| Ordre | 1er | 2eme |
| Steps distilled | 8+3 | 15 (pipeline HQ officiel) |
| Qualite | Bonne | Potentiellement meilleure |
| Speed | Plus rapide (moins de steps) | Plus lent (plus de steps) |
| Usage officiel | DistilledPipeline | TI2VidTwoStagesHQPipeline |

**Verdict** : **Investiguer davantage**. Res2s est le sampler du pipeline HQ officiel, mais avec 15 steps (vs 8+3). Si la qualite justifie les steps supplementaires pour le use case Medusa, ca vaut le coup. Necessite un test A/B sur des images reelles.

---

## 6. Plan d'action recommande

### Priorite 1 — Stabilite + Performance immediate
1. **Pin SageAttention** : Ajouter `git checkout <commit_post_fix_320>` dans le Dockerfile apres le clone. Verifier que le build utilise le code post-fix SM90.
2. **Tester SageAttention 2.2.0 + torch.compile(reduce-overhead)** : Supprimer `torch.compiler.disable(sageattn)`, passer en `mode="reduce-overhead"` inconditionnellement. Si ca fonctionne sur H100, deployer.

### Priorite 2 — Amelioration incrementale
3. **Si P1.2 echoue : mode="max-autotune-no-cudagraphs"** : Fallback qui offre l'autotuning Triton sans CUDA graphs.
4. **Evaluer Res2s** : Test A/B qualite sur 10-20 images avec Euler 8+3 vs Res2s 8+3 (meme nombre de steps). Si qualite superieure, adopter.

### Priorite 3 — Preparation future
5. **Renommer `image_conditionings_by_replacing_latent`** → `combined_image_conditionings` dans pipeline.py pour preparer un eventuel update du pin ltx-pipelines.
6. **Surveiller** les releases Lightricks pour des benchmarks fp8_scaled_mm et des updates ltx-pipelines.

---

## 7. Ce qu'il ne faut PAS changer

| Element | Raison |
|---------|--------|
| **QuantizationPolicy.fp8_cast()** | Fonctionne, pas de dependance externe, qualite BF16 preservee. fp8_scaled_mm necesite tensorrt_llm sans gain prouve |
| **Checkpoint distilled BF16** | Le checkpoint FP8 scaled (`ltx-2.3-22b-dev-fp8.safetensors`) est incompatible avec la fusion LoRA dans ltx-core. Meme si pas de LoRA actuellement, garder la flexibilite |
| **VAE sans tiling** | H100 80GB a assez de VRAM. Tiling ajoute du overhead et un risque de ghosting |
| **Audio desactive** | Pas de use case audio pour Medusa I2V |
| **Embeddings warmup en process isole** | Permet un cleanup VRAM complet avant le handler. Ne pas fusionner dans le process principal |
| **Dedup cache** | Economise du compute pour les requetes repetees, cout negligeable |
| **Post-processing async** | Le GPU travaille pendant que le CPU encode le MP4. Ne pas serialiser |
| **Transformer cache pre-fusionne** | Reduit le cold start significativement. Le hash d'invalidation est correct |
| **Pin ltx-core/ltx-pipelines au commit 9e8a28e** | Version stable testee. Ne pas bumper sans raison — les PRs recentes (#136) renomment des APIs (breaking changes) |
| **Flash Attention 3** | Beta, pas sur PyPI, installation complexe, aucun avantage prouve sur SageAttention 2.2.0 pour ce use case |
| **TensorRT** | Enorme complexite pour un gain non quantifie. torch.compile suffit |

---

## Sources verifiees

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2) — commits, PRs, READMEs
- [LTX-2 PR #136](https://github.com/Lightricks/LTX-2/pulls/136) — sync 2026-03-05, API renames
- [LTX-2 Issue #121](https://github.com/Lightricks/LTX-2/issues/121) — FP8 scaled tensor mismatch
- [SageAttention GitHub](https://github.com/thu-ml/SageAttention) — releases, issue #320 (SM90 fix)
- [SageAttention Issue #320](https://github.com/thu-ml/SageAttention/issues/320) — SM90 backend broken, fixed
- [Flash Attention PyPI](https://pypi.org/project/flash-attn/) — v2.8.3, FA3 separate (hopper dir)
- [PyTorch torch.compile docs](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) — modes comparison
- [ltx-core README](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/README.md) — quantization, loader
- [ltx-pipelines README](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md) — pipelines, fp8 options
- Codebase source : pipeline.py, handler.py, warmup_embeddings.py, Dockerfile, start.sh
