# Phase 2 : géolocalisation d'une photo par sa ligne d'horizon

État au 2026-10-08. Tout est calculé localement, et les photos de l'utilisateur ne quittent pas la machine.

## Mise à jour 0.2 (2026-10-16) : ~18x plus rapide, test CrossLocate

Mêmes résultats, ~18x plus vite sur un GPU : localisation en mode index ~1,45 s -> ~0,08 s par photo (RTX 5090).

| Étape (par photo, 1 GPU) | 0.1 | 0.2 | Comment |
|---|---|---|---|
| index (présélection) | 0,46 s | 0,03 s | index sur GPU sans les mots jamais lus (64 % des entrées), vote en un noyau Triton, votes par tuiles qui tiennent dans le cache L2 |
| raffinement | 0,73 s | 0,04 s | horizons calculés dans la fenêtre de la photo ± 15°, lancer de rayons en un noyau Triton |
| reste | ~0,25 s | ~0,01 s | photos préchargées |

- **Exactitude.** L'index propose les mêmes lieux (seul l'ordre des égalités change). Le lancer de rayons fusionné
  compte les positions depuis le coin du MNT : en float32, les coordonnées LV95 (~2,6e6 m) ne sont justes qu'à
  ~25 cm, ce qui faisait bouger un relief à moins de 100 m de la caméra de jusqu'à ~0,8°. Contre une référence en
  float64 : écart moyen 0,018° (0.1) -> 0,003° (0.2).
- **Fenêtre du raffinement.** À ± 5°, le taux de réussite ne bougeait pas mais la position se dégradait un peu
  (smapshot : 40 -> 43 m en médiane, cap max 7 -> 9,9°) ; à ± 15°, on retrouve le raffinement sur 360° pour 4 % du
  coût en plus.

| Jeu de test | brute 0.1 | brute 0.2 | index 0.1 | index 0.2 |
|---|---|---|---|---|
| CH1 | 88,2 % | 88,7 % | 87,7 % | 88,2 % |
| smapshot test | 73,4 % | 73,4 % | 71,7 % | 71,7 % |
| test CrossLocate (3 513 photos) | — | — | — | 59,8 % |

Photo par photo, la 0.2 tombe au même point que la 0.1 pour l'immense majorité des photos ; les écarts restants
sont équilibrés (smapshot brute : 9 photos plus près de la vérité, 8 plus loin ; erreur médiane 40 m inchangée).

**Test CrossLocate** (Tomešek et al., WACV 2022, 38,7 % annoncés) : 59,8 % à 1 km au premier rang, 65,7 % dans les 20
premiers ; Flickr 77,0 %, LandscapeAR 49,9 % (poses par reconstruction 3D : la vraie position n'est dans la
présélection que pour 69 % d'entre elles), CH1 90,8 %. Erreur médiane 219 m, cap 0,85° quand c'est juste. Leur
`yaw` vaut ~269,6° - cap grille. Aucune photo de test dans notre entraînement (scripts/overlap_check.py).

## TL;DR

- **CH1** (203 photos de référence, ETH) : **88,2 % localisées à moins de 1 km au premier essai**.
  - C'est le niveau de Baatz et al. 2012 (88 %), qui corrigeaient en partie la skyline à la main. Ici, tout est automatique.
- **Test smapshot** (459 photos prises du sol, zones jamais vues à l'entraînement) : **73,4 %**.
  - Quand la photo est bien localisée : 40 m d'erreur médiane, 0,5° d'erreur de cap.
- **Index de mots de crête** : même précision (87,7 % sur CH1) en ~1 s par photo, au lieu de 3,8 s pour la recherche exhaustive.
  - Le coût de la recherche ne grandit presque plus avec la surface couverte.
- **Photos des Fiz** (sans GPS) :
  - IMG_3455 est retrouvée au **lac d'Anterne**, confirmé par l'utilisateur ;
  - IMG_3373 / 3374 tombent à Plaine-Joux, à confirmer ;
  - les autres restent ratées (skyline dans les nuages, ou trop courte).
- **Hors Suisse**, les MNT mondiaux font chuter les résultats (~68 % sur CH1, ~27 % sur smapshot, avec Copernicus comme avec FABDEM).
  - Cause : leurs erreurs de hauteur sur les arêtes et les parois, de 20 à 40 m (MNT radar). Ni leur résolution de 30 m, ni la végétation ne l'expliquent. Voir le §5.
  - Le passage à l'échelle passera par des MNT lidar nationaux.

## 1. Chaîne de traitement

1. **Skyline de la photo.** Crestline (U-Net ConvNeXt, publié sur HF) donne la probabilité de ciel. La skyline est la première ligne non-ciel de chaque colonne.
   - Les colonnes peu fiables sont écartées : confiance « skyline » de Crestline < 0,6, lissée sur 15 colonnes. C'est typiquement le bord des nuages.
2. **Profil angulaire.** Avec la focale (EXIF, ou fournie), chaque colonne donne un azimut relatif et une élévation.
   - Le profil est calculé pour 11 hypothèses de tangage, de -12° à +18°.
3. **Base d'horizons** (`scripts/horizon_db.py`).
   - Grille de 125 m sur l'emprise suisse (y compris la zone des Fiz) : 4,96 M points de vue, caméra à 1,7 m du sol.
   - Pour chaque point : élévation de l'horizon sur 360°, par pas de 0,25° (1 440 valeurs).
   - Lancer de rayons sur GPU jusqu'à 120 km, avec courbure terrestre et réfraction. MNT : swissALTIRegio, en pyramide 10/20/40/80 m.
   - Construction : 5,5 min sur 3 RTX 5090. Taille : 27 Go.
4. **Comparaison** (`scripts/geolocate.py`).
   - Écart quadratique entre les profils, minimisé sur le décalage vertical (tangage résiduel) et sur une pente bornée (roulis ±6°).
   - La FFT teste tous les caps d'un coup, pour tous les points.
5. **Raffinement.** Les 100 meilleurs candidats sont re-testés sur une grille de 25 m (±100 m), avec des horizons recalculés.
6. **Index de mots de crête** (`scripts/ridge_index.py`), qui remplace l'étape 4 exhaustive.
   - Les panoramas sont découpés en mots : morceaux de 10° et de 4°, 8 valeurs quantifiées en 5 niveaux, relatives au centre du morceau.
   - Les mots sont rangés dans un index inversé.
   - Les mots de la photo votent pour (lieu, cap). Les 1 000 lieux les plus votés passent à la comparaison exacte, puis au raffinement.
   - Index : 3,2 milliards d'entrées, 18 Go, ~7 min de construction sur CPU.

## 2. Résultats (top-1 = meilleur candidat à moins de 1 km de la vérité)

| Version | CH1 top-1 | CH1 top-20 | smapshot test top-1 | Temps / photo |
|---|---|---|---|---|
| Grille 250 m, profil brut | 42 % | 61 % | — | |
| + roulis (pente bornée) | 79 % | 89 % | 49 % | |
| + raffinement local | 83 % | 90 % | 61 % | |
| **Grille 125 m + raffinement (exhaustif)** | **88,2 %** | 95,1 % | **73,4 %** | 3,8 s |
| Index + vérification | 87,7 % | 92,1 % | 71,5 % | ~1 s |
| Index + filtre de colonnes fiables | 87,7 % | 91,1 % | 71,7 % | ~1 s |

- Avec l'index, la vraie position est dans la présélection de 1 000 lieux dans 94 % des cas.
- Le filtre de colonnes est réglé sur le val smapshot (+2,8 points). Il est neutre sur les jeux de test, mais décisif sur les photos nuageuses de l'utilisateur.

## 3. Ce qui n'a pas marché (et pourquoi)

- **A priori sur le tangage centré sur 0°.** Il dégrade les résultats, car les photos de montagne visent souvent 10-16° vers le haut. Retiré.
- **Critère de netteté ciel/terrain.** Il retire trop de colonnes valides (54 % au lieu de 71 %). Écarté.
- **Grille de 250 m.** La moitié des échecs smapshot étaient des ratés de recherche : le relief proche change trop vite. À 250 m de la vraie position, le profil est déjà méconnaissable. Corrigé par la grille de 125 m et le raffinement.

## 4. Vitesse

- Le score exhaustif est limité par la bande passante mémoire du GPU : 75 % du temps part dans des opérations élément par élément.
- **Base répartie sur 3 GPU** + calcul commun à tous les tangages + suppression des synchronisations : ×6, de 23 s à 3,8 s par photo.
- **Index** : ~0,15 s pour la présélection, plus ~0,8 s pour le raffinement, dont le coût est fixe.

## 5. Passage à l'échelle hors Suisse : quel MNT ?

Mêmes scripts (`MARCO_DEM=...`), grille de 125 m, index + filtre :

| MNT | CH1 top-1 | smapshot test top-1 |
|---|---|---|
| swissALTIRegio 10 m (sol nu) | 87,7 % | 71,7 % |
| swissALTIRegio dégradé à 30 m (sol nu) | 85,7 % | 63,9 % |
| Copernicus GLO-30 (surface : arbres et bâtiments compris) | 67,5 % | 26,4 % |
| FABDEM 30 m (Copernicus sans arbres ni bâtiments) | 68,0 % | 27,8 % |

**Diagnostic.**
- **Pas la résolution** : swisstopo dégradé à 30 m garde 85,7 %.
- **Pas la végétation** : FABDEM, sans arbres ni bâtiments, fait comme Copernicus.
- **Pas un décalage de reprojection** : la meilleure translation trouvée est de 0 m.
- **C'est l'erreur de hauteur sur les reliefs raides.** 90ᵉ centile de l'écart à swisstopo :

| Zone | FABDEM | swisstopo dégradé à 30 m |
|---|---|---|
| Zermatt | 18 m (25 m en pente forte) | 4 m |
| Aletsch | 44 m | 4 m |

Copernicus est un MNT radar (TanDEM-X), réputé imprécis sur les fortes pentes. Une erreur de 30 m sur une crête à 1 km fait ~1,7°, pour un matching à ~0,15° près.

Avec FABDEM, la vraie position n'est plus dans la présélection de l'index que dans 40 % des cas (contre 77 % avec swisstopo). La vérification exacte rate aussi davantage.

**Essai sans succès : pondérer chaque direction par l'incertitude attendue du MNT.**
- Principe : σ_angle ≈ σ_z / distance de l'horizon, avec la vraisemblance gaussienne complète.
- Résultat sur 200 photos du val smapshot, top-1 :

| MNT | sans pondération | avec pondération |
|---|---|---|
| FABDEM | 25,0 % | 22,5 % (σ_z = 10 m), 21,5 % (20 m), 15,0 % (40 m) |
| swisstopo | 73,0 % | 61,5 % (3 m), 66,5 % (8 m) |

Code retiré.

**Conséquence pour le passage à l'échelle** : l'algorithme passe à l'échelle (index), mais le MNT est le facteur limitant. Il faut des MNT lidar ou photogrammétriques précis, et beaucoup de pays de montagne en publient en libre :
- France : RGE ALTI 1-5 m ;
- Autriche : 1-10 m selon les Länder ;
- Italie : par région ;
- Norvège : 1 m ;
- Espagne : 2-5 m ;
- États-Unis : 3DEP 1-10 m.

Les MNT mondiaux (Copernicus, FABDEM) ne servent que de complément, pour l'horizon lointain.

## 6. Fichiers

| Fichier | Rôle |
|---|---|
| `scripts/horizon_db.py` | base d'horizons (`MARCO_DEM` pour changer de MNT) |
| `scripts/geolocate.py` | profils, comparaison FFT, raffinement, `localize`, `localize_indexed`, filtre de colonnes |
| `scripts/ridge_index.py` | index de mots de crête (`RidgeIndex`, `ShardedIndex`) |
| `scripts/eval_geoloc_ch1.py`, `scripts/eval_geoloc_smapshot.py` | évaluations exhaustives |
| `scripts/eval_geoloc_fast.py` | évaluations avec index, plusieurs réglages à la fois |
| `scripts/geolocate_photos.py` | photos sans GPS (focale EXIF) |
| `scripts/viz_geoloc.py` | planche de diagnostic (profil photo contre profil MNT) |
| `scripts/copernicus_dem.py`, `scripts/degrade_dem.py` | MNT alternatifs reprojetés sur la grille suisse |

Les rapports sont dans `outputs/geoloc_*/report.json`, les sorties sur les Fiz dans `outputs/fiz_geoloc_v2/`.
