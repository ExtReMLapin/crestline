# Reproduire les recherches (entraînement de Crestline, évaluations)

POC : extraire la ligne d'horizon (skyline) et les crêtes internes d'une photo, puis les comparer à des
profils synthétiques calculés depuis un MNT pour retrouver où la photo a été prise.

- État de l'art : `docs/etat_de_l_art.md`

## Environnement

```bash
uv sync    # Python 3.12, torch 2.14 + CUDA 13.0 (RTX 5090 / Blackwell), transformers, rasterio, pyproj...
```

Les photos personnelles (`photos/`) ne sont jamais envoyées en ligne. Ne sont téléchargées que des
données publiques : MNT swisstopo, photos smapshot (ETH-Bibliothek / swisstopo), GeoPose3K, CH1.

## Données

| Donnée | Où | Comment |
|---|---|---|
| swissALTIRegio 10 m (CH + bords FR/IT) | `data/dem/swissaltiregio_2056_5728.tif` | `wget https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio/swissaltiregio_2056_5728.tif` |
| swissALTI3D 2 m (tuiles 1 km) | `data/dem/swissalti3d/` | `python scripts/alti3d_prefetch.py index`, puis `fetch <selection.csv>` |
| Photos smapshot + poses | `data/smapshot/` | `scripts/smapshot_catalog.py`, `scripts/smapshot_photo_prefetch.py` |
| GeoPose3K | `data/geopose3k/` | archive de 40 Go, voir `docs/etat_de_l_art.md` |
| CH1 (skylines manuelles, ETH) | `data/ch1/` | Wayback Machine, `CH1.tgz` |

## Pipeline d'étiquettes smapshot (la voie retenue)

```bash
# 1. catalogue + sélection (découpage géographique train/val/test par mailles de 10 km)
uv run python scripts/smapshot_catalog.py data/smapshot/catalog.csv
# 2. tuiles MNT 2 m et photos (téléchargements parallèles)
uv run python scripts/alti3d_prefetch.py index
uv run python scripts/alti3d_prefetch.py fetch data/smapshot/selection.csv 32
uv run python scripts/smapshot_photo_prefetch.py data/smapshot/selection.csv data/smapshot/out 12
# 3. étiquettes : rendu MNT (GPU), affinage d'orientation, masques à ignorer (2 processus par GPU conseillés)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/smapshot_labels.py <liste_ids.txt> data/smapshot/out 1024
# 4. revue visuelle
uv run python scripts/show_smapshot_labels.py data/smapshot/out apercu_entrainement/smapshot_revue
# 5. listes filtrées par qualité + cache d'entraînement
uv run python scripts/smapshot_splits.py data/smapshot/out data/smapshot/selection.csv data/smapshot/lists
uv run python scripts/build_cache.py data/smapshot/out/labels data/smapshot_cache 40
```

## Entraînement et évaluation

```bash
# un run par GPU (pas de NVLink : les expériences indépendantes passent mieux à l'échelle)
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc_per_node=1 scripts/train_ridges.py \
    --cache data/smapshot_cache --train-list data/smapshot/lists/train.txt --val-list data/smapshot/lists/val.txt \
    --epochs 30 --compile --out outputs/<run>
# run final rapide sur les 3 GPU : --nproc_per_node=3 --compile --grad-bf16 --batch 16
uv run python scripts/eval_smapshot.py data/smapshot/lists/test.txt data/smapshot/out/labels outputs/eval_smapshot outputs/<run>/best.pt
uv run python scripts/eval_ch1.py data/ch1/extracted outputs/eval_ch1 outputs/<run>/best.pt
uv run python scripts/predict_photos.py outputs/<run>/best.pt photos outputs/<run>/fiz_2048 2048
```

## Géolocalisation : évaluations

Base d'horizons suisse (`data/horizon/ch125_band*`) et index (`data/horizon/index125_band*`) : voir
[PHASE2_geolocalisation.md](PHASE2_geolocalisation.md).

```bash
# test « Uniform » de CrossLocate (Tomešek et al., WACV 2022) : métadonnées + photos de test seulement (~4 Go)
B=https://cphoto.fit.vutbr.cz/crosslocate/dataset
curl -O $B/meta_structures/uniform_dataset/Alps_photos_to_depth_compact_test.npy
uv run python -I scripts/crosslocate_meta.py Alps_photos_to_depth_compact_test.npy data/crosslocate/test_queries.json
# (extraire de query_photos.tar.gz les seules photos listées dans test_queries.json, en flux : tar -T)
# aucune photo de test ne doit figurer dans l'entraînement de Crestline (verdict final à l'œil)
uv run python scripts/overlap_check.py data/crosslocate/query_photos data/smapshot/lists_all/train.txt

# tout : mode index (une photo sur trois par GPU, 3 GPU en parallèle) puis mode brute, ~35 min
scripts/validate_all.sh v3
```

## Scripts

| Script | Rôle |
|---|---|
| `geolocate.py`, `horizon_db.py`, `ridge_index.py` | géolocalisation : profils, matching FFT, raffinement, base d'horizons, index |
| `horizon_fused.py` | lancer de rayons en un seul noyau GPU (Triton) |
| `eval_geoloc_ch1.py`, `eval_geoloc_smapshot.py` | évaluations en mode brute (3 GPU) |
| `eval_geoloc_fast.py`, `eval_all_parallel.sh`, `validate_all.sh` | évaluations en mode index (CH1, smapshot, CrossLocate), en parallèle |
| `crosslocate_meta.py`, `overlap_check.py` | test CrossLocate : métadonnées (lecture sûre du pickle), recouvrement avec l'entraînement |
| `dem_render.py` | rendu GPU de cartes de distance depuis le MNT suisse (courbure, réfraction, 3 niveaux) |
| `smapshot.py` | API smapshot, modèle de caméra (vérifié au pixel), téléchargement IIIF / Deep Zoom |
| `smapshot_labels.py` | étiquettes ciel / skyline / crêtes + masques à ignorer, affinage d'orientation |
| `train_ridges.py`, `ridge_data.py` | U-Net ConvNeXt, 3 sorties (ciel, skyline, crêtes), cache mmap uint8 |
| `eval_smapshot.py`, `eval_ch1.py`, `eval_ridges.py` | évaluations (smapshot test, CH1 manuel, GeoPose3K) |
| `geopose_labels.py`, `prepare_geopose.py`, `realign_labels*.py` | voie GeoPose3K (abandonnée comme source principale) |
| `exif_report.py`, `predict_photos.py` | photos personnelles : EXIF, prédiction |
