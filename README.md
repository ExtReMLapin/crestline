# Crestline

*S.D.E.C.E. : Système de Détection de l'Emplacement par Comparaison des Escarpements — nom pris au hasard.*

**Retrouver d'où une photo de montagne a été prise, sans GPS, à partir de sa ligne d'horizon.**

![Lac d'Anterne : skyline de la photo (vert) et skyline calculée depuis le relief au point trouvé (cyan)](docs/images/demo_anterne.jpg)

*Photo du lac d'Anterne (Haute-Savoie). Elle est retrouvée à sa position et à son cap (343°) parmi 230 000
points de vue. En cyan, la ligne d'horizon calculée depuis le relief au point trouvé.*

## L'histoire

J'ai ce projet en tête depuis 2021 : la ligne des crêtes contre le ciel est une empreinte, et vue d'un endroit
donné elle est unique. En la comparant à toutes les lignes d'horizon calculées depuis un modèle de terrain, on
doit pouvoir retrouver d'où une photo a été prise.

Je ne m'y suis jamais mis, faute de temps : entre l'état de l'art, la fabrication d'un jeu d'entraînement, un
réseau de neurones, un moteur de rendu de relief sur GPU et un moteur de recherche, ça représentait des mois de
travail pour un projet perso, donc ça n'en valait pas la peine.

En 2026, je l'ai fait avec [Claude Code](https://claude.com/claude-code) en quelques jours :

- la recherche bibliographique ;
- la fabrication d'annotations à partir de photos recalées par des bénévoles ;
- l'entraînement et la publication du modèle ;
- la base d'horizons de toute la Suisse ;
- l'index de recherche ;
- les évaluations ;
- ce paquet.

Ce qui n'était pas rentable à développer l'est devenu.

## Essayer tout de suite

```bash
pip install git+https://github.com/ExtReMLapin/crestline
crestline demo
```

La démo prend quelques minutes sur un GPU NVIDIA :

1. elle télécharge une partie du relief suisse (quelques centaines de Mo) ;
2. elle construit une carte de 60 × 60 km autour du Mont-Blanc ;
3. elle retrouve la photo du lac d'Anterne.

Installez d'abord le [torch adapté à votre carte](https://pytorch.org/get-started/locally/) ; les RTX 50xx
demandent CUDA ≥ 12.8.

### Une photo trouvée sur Internet, en une commande (Cervin, Suisse)

```bash
crestline locate https://upload.wikimedia.org/wikipedia/commons/b/bd/Weissgrat_from_Gornergrat_2021_04.jpg --near "Gornergrat" --radius 10
```

```
Gornergrat : Gornergrat, Zermatt, Visp, Valais (45.9833, 7.7850), rayon 10 km
source du relief : swisstopo
zone 20 × 20 km, 26 k points de vue : téléchargement ~0.1 Go, disque ~0.3 Go, mémoire GPU ~0.4 Go, ~1 min
...
Weissgrat_from_Gornergrat_2021_04.jpg : position GPS trouvée dans l'EXIF (45.98335, 7.78503). Elle n'est pas
utilisée pour chercher, on lance quand même ; l'écart sera affiché à la fin.
Weissgrat_from_Gornergrat_2021_04.jpg : 45.98347, 7.78520  cap 97°  score 0.106°  confiance élevée
  OSM    : https://www.openstreetmap.org/?mlat=45.98347...
  écart avec le GPS de l'EXIF : 19 m
```

Voici ce qui se passe, en ~1 min la première fois et en quelques secondes ensuite :

1. `--near` situe la zone grâce à OpenStreetMap.
2. Le relief le plus précis disponible est choisi automatiquement : IGN en France, swisstopo en Suisse.
3. La carte est construite, puis gardée en cache.
4. La focale est lue dans l'EXIF de l'image.

Le GPS de la photo sert seulement à afficher l'écart à la fin. Photo : Wikimedia Commons, CC BY-SA 4.0.

### Construire une carte une fois, pour beaucoup de photos

```bash
crestline map build cartes/zermatt --near "Zermatt" --radius 12 --index
crestline map build cartes/anterne --source ign --bbox 6.70,45.93,6.90,46.05 --index
crestline locate https://raw.githubusercontent.com/ExtReMLapin/crestline/main/crestline/demo/lac_anterne.jpg --map cartes/anterne
```

Le RGE ALTI (IGN, lidar, 5 m) est téléchargé tuile par tuile depuis la Géoplateforme. Un téléchargement
interrompu reprend là où il s'était arrêté.

- **Estimation avant de lancer** : avant tout gros travail, l'outil affiche le volume à télécharger, l'espace disque, la mémoire GPU et le temps estimés, puis demande confirmation (`--yes` pour ne pas la demander).
- **Longitude négative** (Pyrénées, ouest de la France) : écrivez `--bbox=-0.30,42.62,0.02,42.86`, sinon le `-` est pris pour une option.

### Vos photos

```bash
crestline locate photo.jpg --map cartes/anterne
crestline locate mes_photos/ --map cartes/anterne --out resultats/
crestline locate https://exemple.org/photo.jpg --map cartes/anterne --fov 65   # focale absente de l'EXIF
```

## Sources de relief

**La précision du relief compte plus que tout le reste.** Une erreur de 30 m sur une arête située à 1 km fait
~1,7°, alors que la comparaison travaille à ~0,15° près.

`--source auto` (par défaut avec `--near`) choisit pour vous.

| `--source` | Couverture | Précision | Licence | Conseil |
|---|---|---|---|---|
| `ign` | France | lidar, 5 m (1 m disponible) | Licence Ouverte Etalab 2.0 | **recommandé en France** |
| `swisstopo` | Suisse + ~100 km autour | 10 m (2 m en Suisse) | libre, « © swisstopo » | **recommandé en Suisse** |
| `copernicus` | monde | radar, 30 m | libre (ESA) | dépannage : ~68 % de réussite au lieu de 88 % |

- **Ce qui est téléchargé** : la zone des points de vue + 30 km à la maille fine (relief proche), et la zone + 120 km à 80 m (horizon lointain).
- **Les trous** sont comblés par la source suivante (`ign` → `swisstopo` → `copernicus`), et la provenance est notée dans la carte.
- **Réseau poli** : chaque installation s'identifie auprès des services avec un identifiant aléatoire (`~/.config/crestline/install_id`, sans rien révéler de la machine). Le géocodage respecte la règle d'OpenStreetMap (1 requête par seconde) et ses résultats sont gardés en cache.
- **Rien n'est fait deux fois.** Relancer la même commande réutilise le relief déjà téléchargé, les horizons et l'index (même source, même maille, zone contenue dans la précédente). Si un réglage change (pas, hauteur, zone), seule la carte est recalculée. `--force` refait tout.
- **Pour juste récupérer le relief** : `crestline dem fetch --source ign --bbox … --out dossier/`.
- **Votre propre MNT** (GeoTIFF, n'importe quel système de coordonnées) : `--dem mon_mnt.tif` au lieu de `--source`.

## Utilisation

### Deux modes de recherche

```
 MODE BRUTE                                  MODE INDEX
 ──────────                                  ──────────
 map build                                   map build --index  (ou : map index CARTE)
 compare la photo à TOUS les points          l'index propose ~1 000 lieux, comparés exactement
 + rien d'autre à construire                 + ~1 s par photo, presque indépendant de la surface
 - coût proportionnel à la surface           - rate la bonne réponse ~5 % du temps
 → petite zone                               → région ou pays
```

Par défaut (`--mode auto`), l'index est utilisé s'il existe.

### Sortie

Par défaut, les résultats vont dans `crestline_out/<horodatage>_<photo>/` ; on peut choisir un autre dossier
avec `--out`.

```
result.json        top 5 : lat/lon, cap, tangage, roulis, score, confiance, liens OpenStreetMap / Google Maps
result.geojson     points + direction de visée (geojson.io, QGIS)
summary.md         résumé lisible
skyline.jpg        skyline détectée : vert = utilisée, rouge = ignorée (bord de nuage...)
candidates.jpg     planche des 5 candidats
candidates/N.jpg   le relief rendu depuis le candidat N, superposé à la photo
```

La **confiance** compare le score du 1er lieu à celui du 2e. Une bonne réponse se détache nettement ; quand
plusieurs lieux se valent, la confiance est « faible » et le résultat est à prendre avec prudence.

### Configuration

| Réglage | Option | Défaut |
|---|---|---|
| Source du relief | `--source ign\|swisstopo\|copernicus`, ou `--dem fichier.tif` | — |
| Maille du relief téléchargé | `--cell` | 5 m (ign), 10 m (swisstopo), 30 m (copernicus) |
| Zone de la carte | `--near "lieu" --radius 20`, `--bbox lon0,lat0,lon1,lat1`, ou `--bounds x0,y0,x1,y1` | tout le MNT |
| Système de coordonnées | `--crs EPSG:2154` (reprojection automatique) | celui du MNT ; UTM si le MNT est en degrés |
| Pas de grille | `--step` | 125 m |
| Hauteur de la caméra | `--camera-height` (enregistrée dans la carte) | 1,7 m |
| GPU | `--device cuda:0,cuda:1`, ou `--device cpu` | tous les GPU |
| Modèle | `--model` (dépôt HF ou dossier local), ou `$CRESTLINE_MODEL` | `LPN64/crestline-convnext-base` |
| Cache des poids | `$HF_HOME` | `~/.cache/huggingface` |
| Cache (démo, cartes `--near`, géocodage) | `$CRESTLINE_CACHE` (`--cache` pour la démo) | `$XDG_CACHE_HOME/crestline` |
| Confirmation des gros travaux | `--yes` | demandée au-delà de ~10 min, 2 Go téléchargés ou 10 Go sur disque |
| Sortie | `--out`, `--top`, `--no-render` | `crestline_out/…`, 5 candidats |
| Filtre de la skyline | `--min-conf` (0 : tout garder) | 0,6 |

### Focale

La focale réelle (zoom compris) est lue dans l'EXIF. Pour la convertir en pixels, l'outil utilise soit
l'équivalent 35 mm, soit la densité de pixels du capteur, ce qui fonctionne aussi sur une photo redimensionnée.
Si l'EXIF n'a rien, il faut la fournir avec `--fov` (champ horizontal, en degrés) ou avec `--focal-px`.

## Comment ça marche

```
   PHOTO                                              MNT
     │                                                 │
     ▼                                                 ▼
 Crestline (U-Net ConvNeXt)                   horizon sur 360° de chaque point
 skyline + confiance par colonne              d'une grille de 125 m, caméra à 1,7 m
     │                                                 │
     ▼                                                 ▼
 silhouette angulaire ────── comparaison (FFT : tous les caps d'un coup) ──────┐
 (focale, 11 tangages)        décalage vertical + pente (roulis) absorbés      │
                                       │                                       │
                                       ▼                                       │
                         100 meilleurs lieux ──▶ raffinement sur 25 m ──▶ top 5
```

- **Crestline** est entraîné sans annotation manuelle. On a pris des photos dont la pose a été recalée par les bénévoles de [smapshot](https://smapshot.heig-vd.ch), puis on a calculé les crêtes par lancer de rayons dans le MNT suisse à 2 m. Détails sur la [fiche du modèle](https://huggingface.co/LPN64/crestline-convnext-base).
- **L'horizon d'un point de vue** est, pour chaque direction, l'angle le plus haut sous lequel on voit le relief, jusqu'à 120 km, courbure terrestre et réfraction comprises.
- **L'index** découpe chaque panorama en « mots de crête » (formes de 10° et de 4° de large). Les mots de la photo votent pour un lieu et un cap : un vrai lieu accumule des votes cohérents.

Le rapport complet se trouve dans [docs/PHASE2_geolocalisation.md](docs/PHASE2_geolocalisation.md).

## Résultats

Une photo compte comme bien localisée si le meilleur candidat est à moins de 1 km de la vérité.

| Jeu de test | Mode brute | Mode index |
|---|---|---|
| CH1 (203 photos des Alpes suisses, ETH Zurich) | **88,2 %** | 87,7 % |
| smapshot (459 photos prises du sol, zones jamais vues) | **73,4 %** | 71,7 % |

- Quand la photo est bien localisée, l'erreur médiane est de 40 m et l'erreur de cap de 0,5°.
- La référence publique sur CH1 (Baatz et al., 2012) est de 88 %, avec des skylines en partie corrigées à la main.

## Limites

- **Une photo seule prise en fond de vallée est un cas limite** : versants proches, peu de sommets lointains. Sur un panorama des Pyrénées, chaque vue de 70° échoue, alors que la silhouette complète du panorama retrouve la position à 170 m.
- **La qualité du MNT est décisive.** Avec les MNT mondiaux (Copernicus, FABDEM, 30 m), CH1 tombe à ~68 %. Leurs erreurs de 20 à 40 m sur les arêtes suffisent à brouiller la silhouette. Utilisez des MNT lidar nationaux (`--source ign`, `--source swisstopo`, ou votre fichier).
- Il faut une skyline visible : ciel bouché, sommets dans les nuages ou cadrage serré sur une paroi font échouer la localisation.
- La caméra est supposée près du sol (randonneur). Les vues aériennes ne sont pas gérées.
- Une carte de la taille de la Suisse au pas de 125 m pèse ~27 Go, et ~18 Go de plus avec l'index.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

Le test tourne sur CPU, sans téléchargement : il fabrique un relief synthétique, rend une photo depuis un point
connu, et vérifie que les deux modes retrouvent ce point et son cap.

## Recherche

L'entraînement de Crestline et les évaluations se trouvent dans `scripts/`, à installer avec
`pip install -e ".[research]"`. Voir [docs/RECHERCHE.md](docs/RECHERCHE.md).

## Licences et attributions

- **Code** : Apache-2.0.
- **Modèle Crestline** : CC BY-SA 4.0 (voir sa fiche).
- **Relief** : swissALTIRegio © swisstopo ; RGE ALTI © IGN (Licence Ouverte Etalab 2.0) ; Copernicus DEM © DLR e.V. 2010-2014 et © Airbus 2014-2018, ESA.
- **Photo de démo** (lac d'Anterne) : © l'auteur, CC BY 4.0.
- Les poses d'entraînement viennent de smapshot (HEIG-VD, ETH-Bibliothek, swisstopo) et de ses contributeurs bénévoles.
- L'évaluation utilise CH1 (Baatz, Saurer, Köser, Pollefeys, ETH Zurich).
