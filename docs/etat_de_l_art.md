# État de l'art — géolocalisation de photos de montagne par matching de lignes de crête

*Rédigé le 2026-10-07. Zone cible du POC : Tour des Fiz (Haute-Savoie, ~46,00 N / 6,72 E), face au massif du Mont-Blanc.*

## Légende de fiabilité

| Marque | Sens |
|---|---|
| **[V]** | Vérifié : page, PDF ou API consultés le 2026-10-07. Lien en HTTP 200, chiffres lus dans le papier, taille de fichier obtenue par requête HEAD. |
| **[A]** | Lu dans le résumé seulement (OpenAlex, page éditeur). Les chiffres détaillés ne sont pas vérifiés. |
| **[M]** | De mémoire ou de seconde main, non vérifié. À confirmer avant de citer. |
| **[D]** | Ma déduction ou mon avis d'ingénieur. Pas un fait publié. |

Pour chaque source : **lien**, **ce qui marche**, **ce qui ne marche pas**, **réutilisable ?**

---

## 0. Ce qu'il faut retenir (synthèse)

1. **Le matching de ligne d'horizon fonctionne, mais l'extraction automatique de l'horizon est le goulot d'étranglement.** C'est le constat le plus robuste de la littérature.
   - Brejcha (thèse 2021) a réévalué la méthode ETH (Saurer IJCV 2016) [V] :
     - environ **30 % de top-1 à moins de 1 km** avec des horizons corrigés à la main (CH1) ;
     - environ **15 %** avec des horizons segmentés automatiquement.
   - Un masque ciel mal segmenté divise donc la performance par environ 2.
2. **À l'échelle régionale (dizaines de milliers de km²), l'horizon seul est faible.**
   - CrossLocate (WACV 2022) compare des photos à des **cartes de profondeur rendues** depuis le MNT, sur une grille de 500 m [V] :
     - **R@1 = 38,7 %** et **R@100 = 72,6 %** à moins de 1 km ;
     - sur environ 40 000 km² de test, avec 1,98 M d'images en base.
   - La profondeur bat la silhouette et la segmentation sémantique.
   - Notre zone est environ 30 fois plus petite. La baseline « horizon seul » y est donc beaucoup plus plausible [D].
3. **La précision métrique exige un a priori de position.**
   - LandscapeAR (ECCV 2020) n'améliore la position que si l'a priori est à moins d'environ 700 m.
   - HorizonNet affine dans une boîte de 80 m.
   - D'où une architecture en deux étapes : **retrieval grossier puis raffinement local** [V/D].
4. **La distribution de l'erreur de cap a une queue lourde** : médiane inférieure à 2°, mais moyenne de 8 à 36° (thèse Brejcha) [V]. Il faut rapporter la médiane **et** des percentiles.
5. **Les modèles récents de géolocalisation globale (GeoCLIP, PIGEON, G3…) sont mauvais en nature.**
   - Sur Im2GPS3k, scènes « naturelles » : 3,3 % à moins de 1 km et 17,5 % à moins de 25 km (Müller-Budack 2018) [V].
   - Ils sont inutiles pour notre zone, déjà connue à quelques dizaines de km près [D].
6. **Les modèles de fondation de 2023 à 2026 n'ont pas encore été publiés sur ce problème.**
   - Aucun papier relu ne fait de matching photo↔rendu MNT avec DINOv2, SAM ou Depth Anything [V-négatif].
   - Ils restent utiles comme **briques** : segmentation du ciel (SAM2, SegFormer ADE20K), crêtes internes par discontinuités de profondeur (MoGe, Depth Anything 3), calibration roulis/tangage (GeoCalib).
7. **Spécificités du Tour des Fiz** [D] :
   - **Les crêtes proches dominent.** Les parois des Fiz sont à quelques centaines de mètres du sentier. Une grille de 250 m et un MNT de 5 m sont insuffisants pour l'horizon proche, qui change très vite avec la position. En contrepartie, c'est très discriminant lors d'un raffinement fin.
   - **L'horizon lointain sort de France.** Mont-Blanc à environ 22 km (côté italien), Dents du Midi à environ 24 km, Grand Combin à environ 45 km, Valais à plus de 70 km. Le MNT de rendu doit déborder largement la zone des points de vue et traverser deux frontières.
   - **Le masque ciel échouera sur certains cas** : neige contre ciel blanc, nuages accrochés aux sommets, brume qui fond les crêtes lointaines. Sur SkyFinder, la segmentation du ciel par temps de neige ne fait que 51 mIoU [V].

---

## 1. Papiers

### 1.1 Références déjà connues, avec chiffres consolidés

#### Baatz et al. (ECCV 2012) et Saurer et al., « Image Based Geo-localization in the Alps » (IJCV 116(3), 2016)
- **Lien :** IJCV doi:10.1007/s11263-015-0830-0. Dataset historique : https://cvg.ethz.ch/research/mountain-localization/ — **en 404 aujourd'hui** [V].
- **Méthode :** « contourlets » de skyline, sac de mots visuels, vote position + direction, vérification des 1000 meilleurs candidats. Zone de 40 000 km² (Suisse), MNT swisstopo d'environ 2 m.
- **Ce qui marche :** le chiffre souvent cité est d'environ 88 % de top-1 à moins de 1 km sur CH1, avec skylines corrigées par l'utilisateur [M, PDF inaccessible].
- **Ce qui ne marche pas :**
  - la réimplémentation indépendante de Brejcha n'obtient qu'environ **30 % (horizons manuels)** et **15 % (horizons automatiques)** [V] ;
  - le choix du MNT pèse lourd : 24 m (viewfinderpanoramas) chez Brejcha contre 2 m chez l'ETH [V].
- **Réutilisable :** les idées oui. Pas de code public [V-négatif].

#### Baboud et al. (CVPR 2011)
- **Méthode :** alignement des arêtes de silhouette, y compris internes, sur le MNT, avec une position connue.
- **Ce qui marche :** 28 photos, 86 % de succès, erreur inférieure à 0,2° [V, d'après le tableau de la thèse Brejcha].
- **Ce qui ne marche pas :** il faut connaître la position. C'est de l'orientation, pas de la géolocalisation.
- **Réutilisable :** oui, comme idée pour l'étape de **raffinement** (alignement des crêtes internes).

#### Brejcha & Čadík, GeoPose3K (Image and Vision Computing 66, 2017)
- Voir la section 3 (Datasets). C'est le dataset de référence, avec pose complète.

#### IARPA Finder et PeakVisor / PeakFinder
- IARPA Finder : programme 2011-2015, rapports et papiers dérivés (Tzeng 2013, Chen 2015). Résultats ci-dessous [V, tableau de la thèse Brejcha, mise en page cassée, donc approximatif].

  | Méthode | Zone | Résultat |
  |---|---|---|
  | Tzeng 2013 | 10 000 km² | « 88 % à 1 km » |
  | Hammoud 2013 | 20 000 km² | 49 % à moins de 14 km |
  | Chen 2015 | 10 000 km² | 60 % à moins de 4,5 km |

- PeakFinder et PeakVisor : moteurs propriétaires. PeakFinder n'expose qu'une API d'intégration web. PeakLens (Politecnico di Milano) est commercial [V].
  - **Non réutilisables.**
  - Ils prouvent toutefois que le rendu d'horizon temps réel et l'alignement d'**orientation** à position connue sont des problèmes résolus en pratique [D].

### 1.2 Travaux postérieurs à 2018 (priorité)

#### CrossLocate — Tomešek, Čadík, Brejcha, WACV 2022 ★
- **Liens :** [PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Tomesek_CrossLocate_Cross-Modal_Large-Scale_Visual_Geo-Localization_in_Natural_Environments_Using_Rendered_WACV_2022_paper.pdf), [projet](http://cphoto.fit.vutbr.cz/crosslocate/), [code](https://github.com/JanTomesek/CrossLocate) [V].
- **Méthode :**
  - retrieval d'images ; requêtes 500×500 ramenées à un champ de 60° ;
  - réseau de type VGG, perte triplet ;
  - comparaison photo ↔ rendus MNT de 3 types : segments sémantiques, silhouettes, profondeur.
- **Base de rendus :** grille de 500 m sur environ 250 000 km² d'Alpes, 12 caps par point (pas de 30°, champ de 60°), soit **10,72 M de rendus**.
- **Requêtes :** 12 353 photos.
- **Résultats [V] :**
  - test (3 513 requêtes contre 1,98 M de rendus, modalité profondeur) : **R@1 = 38,66 %**, **R@100 = 72,62 %** à moins de 1 km ;
  - écrase HLoc (horizon + ICP), DELG et HOW. Son R@1 dépasse leur R@100 jusqu'à environ 5 km de tolérance ;
  - **sans focale connue, la perte est faible.**
- **Ce qui ne marche pas :**
  - l'horizon seul et la segmentation sémantique comme modalités ;
  - l'initialisation ImageNet seule ;
  - combiner les trois modalités n'apporte rien.
- **Réutilisable :**
  - code TensorFlow 1.14, **pas de licence** : référence seulement ;
  - **jeux de rendus et de requêtes téléchargeables**, très utiles pour valider notre moteur de rendu.

#### LandscapeAR — Brejcha et al., ECCV 2020 ★
- **Liens :** [PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf), [projet](http://cphoto.fit.vutbr.cz/LandscapeAR/), [code](https://github.com/brejchajan/LandscapeAR) [V].
- **Méthode :**
  - descripteurs locaux cross-domaine (photo ↔ rendu MNT texturé par orthophoto) ;
  - rendu en éventail de 12 vues autour d'une position a priori ;
  - rétroprojection par la profondeur, puis EPnP + RANSAC.
- **Résultats (test GeoPose3K, 516 images) [V] :**

  | Erreur | Position | Rotation |
  |---|---|---|
  | < 100 m | **30 %** | — |
  | < 300 m | **54 %** | — |
  | < 500 m | **63 %** | — |
  | < 1° | — | 39 % |
  | < 3° | — | 60 % |
  | < 5° | — | 65 % |

  - Bat SIFT, HardNet++, D2Net et NCNet.
- **Ce qui ne marche pas :**
  - il faut **un a priori de position (moins de 700 m environ) et une focale connue** ;
  - le contenu à moins de 100 m environ n'est pas fiable face au MNT ;
  - échoue si la texture est grossière (Huascarán).
- **Réutilisable :** licence BUT non commerciale, pile lourde (COLMAP). Bon pour un **raffinement futur**, pas pour la baseline. **Le dataset « Alps Chamonix » contient la zone des Fiz** (voir section 3).

#### Brejcha & Čadík, « Camera Orientation Estimation in Natural Scenes Using Semantic Cues » (3DV 2018)
- **Lien :** https://cphoto.fit.vutbr.cz/semantic-orientation/ [V]. Une réimplémentation de HLoc et les données sont proposées en ZIP.
- **Méthode :** corrélation croisée sphérique de segments sémantiques + arêtes, à **position connue**.
- **Résultats [V] :**
  - GeoPose3K, aire sous la courbe d'erreur d'orientation : 0,78 (contre 0,52 pour les arêtes seules) ;
  - Venturi : 1,92° en moyenne.
- **Ce qui ne marche pas :** HLoc avec un masque ciel automatique (DeepLab) tombe à 0,49, contre 0,77 en horizons synthétiques. Encore la sensibilité à la segmentation.
- **Réutilisable :** oui, le zip HLoc est une référence d'implémentation.

#### Brejcha, thèse « Visual Localization in Natural Environments » (VUT Brno, 2021)
- **Lien :** https://www.fit.vut.cz/study/phd-thesis-file/887/887.pdf [V].
- **Apport :** la meilleure réévaluation indépendante de la méthode ETH.
  - Zone GP1 de 86 000 km², base d'horizons au pas de 0,001°.
  - Erreur de cap : médiane de −0,2 à −1,8°, mais moyenne de 8 à 36° (queue lourde).
  - Le MNT grossier est désigné comme cause probable de la dégradation.
- **Réutilisable :** comme protocole d'évaluation et tableau comparatif.

#### Kubička, « Camera Orientation Estimation using Vision Transformers » (master, VUT Brno, 2026)
- **Lien :** https://hdl.handle.net/11012/260406 [A].
- **Méthode :** transformer avec deux entrées (panorama MNT 360° + photo). **Pas de focale requise.**
- **Résultats :** comparable à l'état de l'art sur GeoPose3K, moins bon sur Venturi. Pas de chiffres dans le résumé. Orientation seulement.

#### Fetroe, thèse Stanford 2019 — skyline LSH
- **Lien :** https://purl.stanford.edu/cz696hm0983 [V].
- **Ce qui marche :** min-hash LSH sur des contourlets, **10 fois plus rapide** qu'un index inversé à précision égale.
- **Ce qui ne marche pas :** **taux de succès maximal de 17 %** sur 196 photos (MNT ASTER).
- **Réutilisable :** code C++ sous CC BY-NC. Leçon : l'indexation ne compense pas un MNT ou une skyline médiocres [D].

#### Groupe Jin Tang / Fan Guo (Central South University), 2020-2024
- **Liens :**
  - IET Image Processing 2022, doi:10.1049/ipr2.12482 ;
  - CMLocate, IET IP 2023, doi:10.1049/ipr2.12883 ;
  - Applied Sciences 2020, doi:10.3390/app10196661 ;
  - IJGI 2021, doi:10.3390/ijgi10080537 [A].
- **Ce qui marche :** base de skylines rendues par MNT puis matching ; **50 sites sur 202,6 km², erreur moyenne de 43 m en 4,5 s** [A].
  - L'article de 2020 traite explicitement des **collines où la skyline est proche et cassée par la végétation**. C'est directement pertinent pour les Fiz.
- **Ce qui ne marche pas :** pas de code ; chiffres de CMLocate absents du résumé.
- **Réutilisable :** idées seulement (codes de chaîne d'angle, heatmap de distance de skyline).

#### Ruby, Carter et al., « Evaluating Ground Imagery for Long-Standoff… Navigation » (Applied Sciences 16, 2026)
- **Liens :** doi:10.3390/app16157397 ; rapport ERDC 2024, doi:10.21079/11681/49217 [A].
- **Ce qui marche :**
  - horizons synthétiques dans un GeoPackage, matching sur smartphone, relief à plus de 1 km ;
  - **une bonne photo : 75 % de positions à moins de 100 m ; une photo moyenne : 38 %** ;
  - **4 photos ou plus dans des directions différentes depuis le même point : plus de 90 % à moins de 100 m.**
- **Ce qui ne marche pas :** dépend de la qualité de l'image et du rapport relief proche / relief lointain.
- **Leçon [D] :** si plusieurs de nos photos ont été prises au même endroit, les combiner aide énormément.

#### Mikolka-Flöry et al., orientation automatique de photos historiques par l'horizon (ISPRS Open J. 2022 ; Remote Sensing 2021)
- **Liens :** doi:10.1016/j.ophoto.2022.100026, doi:10.3390/rs13091705 [A].
- **Ce qui marche :**
  - **estime aussi la focale depuis l'horizon** : à moins de 5 % pour 75 % des images ;
  - sur 2 400 images : 63 % aussi précises qu'une orientation manuelle, 22 % utilisables comme initialisation, 15 % d'échecs complets ;
  - détection de l'horizon (photos monochromes) : 85 % des images à moins de 10 px.
- **Réutilisable :** idée de l'ajustement de focale sur l'horizon complet, puis résection sur les points saillants.

#### Nagy, recalage d'azimut par matching de skyline (PFG 2020)
- **Lien :** doi:10.1007/s41064-020-00093-1 [A].
- **Contenu :** corrige la boussole du téléphone pour la réalité augmentée. Pas de chiffres dans le résumé. Montre que l'estimation du **cap à position connue** est fiable [D].

#### HorizonNet — Grelsson, Robinson, Felsberg, Khan (IPAS 2018, J. Field Robotics 2020)
- **Lien :** arXiv:2608.30471 [V].
- **Ce qui marche :**
  - un CNN estime tangage et roulis, un autre segmente l'horizon ;
  - corrélation dans le domaine de Fourier (filtre MOSSE) contre des horizons MNT, grille de 4 m dans une boîte de 80 m ;
  - **erreur moyenne de 2,47 m**, contre 6,45 m si l'on ignore tangage et roulis.
- **Ce qui ne marche pas :** les structures fines (arbres isolés) ; l'approche PoseNet bout-en-bout « a complètement échoué ».
- **Leçons [D] :** (1) estimer ou balayer tangage et roulis ; (2) la corrélation FFT est le bon outil pour le balayage de cap.

#### Foucher et al. (Lab-STICC, Lorient), géolocalisation maritime par horizon (OCEANS 2025, AAMAS 2026)
- **Lien :** https://ifaamas.org/Proceedings/aamas2026/pdfs/YIUE1134.pdf [V].
- **Ce qui marche :** champ de 50°, descripteurs d'horizon siamois, recherche vectorielle puis filtre particulaire. **38 m d'erreur moyenne, cap jusqu'à 8°, sur 25 km², départ inconnu.**
- **Réutilisable :** même architecture « descripteur + index vectoriel + vérification » que notre plan. Groupe français, contact possible [D].

#### Planétaire et UAV (moins prioritaire)
- **Lunaire et martien :**
  - Tian et al., IEEE TAES 2024 [A] : simulation seulement ;
  - Lee & Cheung, IEEE Aero 2024 [A] : masque d'horizon infrarouge, quelques mètres à quelques dizaines de mètres, orientation à moins de 0,5° (simulé) ;
  - Abdoli et al., SpaceOps 2025 [V] : grille de 40 m sur 12×12 km, un profil par degré d'azimut, classification par CNN.
- **JPL Censible (ICRA 2024) [V] :** citation utile — l'horizon donne une précision « typiquement de l'ordre de 100 m, au mieux 10 m … précieuse pour l'orientation ».
- **UAV :** Chen et al., IROS 2021 (arXiv:2108.03344) [V] : rendus satellite + topographie, environ 2,5 m d'erreur, sans horizon.
- **Contre-exemple utile :** Wu et al., PLOS ONE 2024 [V]. SuperGlue donne des correspondances photo↔MNT « éparses et incorrectes », et la pose finit faite à la main. Les matchers génériques ne franchissent pas facilement l'écart de domaine.

### 1.3 Pourquoi les méthodes échouent (synthèse transversale, sources ci-dessus)

| Cause | Preuve | Pertinence pour les Fiz [D] |
|---|---|---|
| Masque ciel automatique imparfait | Performance divisée par 2 (Brejcha) ; HLoc-DeepLab tombe de 0,77 à 0,49 | **Élevée** : neige, nuages, brume |
| Pas de skyline visible | Beaucoup d'images GeoPose3K (Ahmad 2021) | Moyenne : photos de parois en contre-plongée |
| Relief et végétation proches | Tang 2020 ; LandscapeAR (moins de 100 m peu fiable) ; HorizonNet (arbres) | **Très élevée** : parois des Fiz, forêts sous le col d'Anterne |
| MNT trop grossier | 24 m contre 2 m (Brejcha) ; MNT de 2 m trop grossier (Wu 2024) | Élevée près des points de vue ; faible au loin |
| Focale inconnue ou fausse | La plupart des méthodes la supposent connue ; CrossLocate et Mikolka-Flöry la tolèrent ou l'estiment | Moyenne : EXIF probablement présent |
| Roulis et tangage | HorizonNet : erreur multipliée par 2,6 si ignorés | Élevée : photos à main levée |
| Queue lourde de l'erreur de cap | Médiane < 2°, moyenne 8 à 36° | Rapporter des percentiles |
| Taille de zone contre pas de grille | Grille de 500 m à 1 km : précision kilométrique | Grille de 250 m demandée ; raffinement local nécessaire |
| Nuit, météo | Peu traités | Selon les photos |

---

## 2. Code open source

Vérifications faites par l'API GitHub ou PyPI le 2026-10-07 (licence, dernière activité).

### 2.1 Rendu d'horizon depuis un MNT

| Projet | Lien | Licence / activité | Ce qui marche | Limites | Réutilisable ? |
|---|---|---|---|---|---|
| **HORAYZON** | https://github.com/ChristianSteger/HORAYZON | MIT, mis à jour en 2026-01 [V] | Lancer de rayons Embree très rapide ; **courbure terrestre gérée** ; script d'horizon par point = profil 360° [V] | Pas sur PyPI ; build conda (embree, tbb) pénible sous Windows ; réfraction seulement pour le soleil | **Oui**, comme référence ou accélérateur. Réfraction via un rayon terrestre effectif [D] |
| **GRASS r.horizon** | https://grass.osgeo.org/grass-stable/manuals/r.horizon.html | GPL, actif [V] | Mode point avec pas angulaire et distance max, sortie CSV/JSON ; courbure [V] | Session GRASS ; lent pour des milliers de points ; pas de réfraction | **Oui, comme oracle** pour valider notre moteur de rendu sur quelques points |
| **GDAL viewshed** | https://gdal.org/en/stable/programs/gdal_viewshed.html | MIT [M] | Courbure + réfraction (coefficient 0,857 par défaut) [V] | Produit une visibilité, pas un profil d'azimut | Partiel : reprendre sa formule |
| **Open3D RaycastingScene** | https://github.com/isl-org/Open3D | MIT, v0.20 (2026-09) [V] | Lancer de rayons Embree installable par pip [D] | Maillage du MNT et courbure à faire soi-même | **Oui**, alternative pip à HORAYZON |
| pyrender / PyVista | github.com/mmatl/pyrender, github.com/pyvista/pyvista | MIT [V] | Rendus profondeur et silhouette, visualisation | pyrender peu maintenu, rendu headless difficile sous Windows | Pour les visualisations et la phase « crêtes internes » |
| xarray-spatial, WhiteboxTools, solshade, richdem | — | MIT / MIT / MIT / GPL [V] | Indices raster (angle d'horizon par azimut, champ de vision) | Pas de profil par observateur, ou pas de courbure ; richdem n'a pas de viewshed | **Non** |
| artificial-panos, TerraLab | github.com/lnw/artificial-panos, github.com/ArcadiaLliure/TerraLab | **Sans licence** [V] | Rendus de panoramas | Licence absente | Lecture seulement |

**Formule de courbure + réfraction** (à implémenter nous-mêmes) [V via GDAL, chiffres calculés] :

    Δh = d² / (2R) · (1 − k),   avec R = 6 371 km et k ≈ 0,13

| Distance | Abaissement |
|---|---|
| 25 km | ≈ 43 m (≈ 0,1°) |
| 50 km | ≈ 170 m |
| 80 km | ≈ 435 m |

C'est indispensable pour le Valais vu depuis les Fiz.

### 2.2 Segmentation du ciel et extraction de skyline

| Projet | Lien | Licence | Ce qui marche | Limites | Réutilisable ? |
|---|---|---|---|---|---|
| **SegFormer / Mask2Former / OneFormer (ADE20K, classe « sky »)** | HF `nvidia/segformer-b5-finetuned-ade-640-640`, `facebook/mask2former-swin-large-ade-semantic` | SegFormer : licence NVIDIA, non commerciale ; Mask2Former : MIT [V] | Une ligne avec `transformers`, **inférence locale**. SegFormer-B2 fine-tuné ciel : IoU 92 % (modèle tiers `Realcat/skywater_seg`, MIT, non publié en revue) [V] | Neige, brume, nuages bas | **Oui : premier choix pour la baseline** |
| **SAM2** | https://github.com/facebookresearch/sam2 | Apache-2.0 [V] | Bords nets ; prompt par un point dans le ciel | Sans classe, il faut un prompt | **Oui**, pour affiner le bord |
| SAM 3 | https://github.com/facebookresearch/sam3 | Licence SAM, poids sous accès contrôlé [V] | Prompt texte (« sky », « snow », « cloud ») | Aucune évaluation sur ciel ou relief | Option |
| **skyline_detection (Ahmad)** | https://github.com/TouqeerAhmad/skyline_detection | MIT [V] | Filtres + programmation dynamique (plus court chemin) pour une skyline continue ; liens vers CH1 et GeoPose3K | Étape de programmation dynamique en MATLAB | **Idée de la programmation dynamique à porter en Python** |
| Sky-Segmentation-and-Post-processing | https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing | MIT [V] | Post-traitement des bords | Poids hébergés sur Baidu | Post-traitement seulement |
| YUNet / SkylineDet-YOLOv11Seg | arXiv 2502.12449 ; github.com/kuazhangxiaoai/SkylineDet-YOLOv11Seg | AGPL-3.0 [V] | Erreur de skyline de 1,36 px sur CH1 [V] | AGPL | À éviter (licence) |

### 2.3 Code de papiers et projets de géolocalisation par skyline

| Projet | Lien | Licence | Ce qu'on y trouve | Réutilisable ? |
|---|---|---|---|---|
| CrossLocate | https://github.com/JanTomesek/CrossLocate | **Aucune** [V] | Entraînement du retrieval (TF 1.14) | Référence ; **données** très utiles |
| LandscapeAR | https://github.com/brejchajan/LandscapeAR | BUT, non commercial [V] | Structure from motion contre MNT, descripteurs, moteur de rendu `itr` | Recherche seulement, lourd |
| **SkylineGeolocation** | https://github.com/pxrxp/SkylineGeolocation | **Aucune** [V], 0 étoile, 2026-09 | **Pipeline quasi identique à notre POC** : base d'horizons GLO-30, segmentation U-Net, profils à 720 classes, NCC + fusion de rangs. Évalué sur seulement 68 panoramas Street View (Khumbu) [V] | **Architecture à étudier.** Ne pas copier sans licence |
| horizon-locate | https://github.com/prasiddha-8636/horizon-locate | Non précisée [V] | Même projet Khumbu ou un jumeau : U-Net, profil, NCC contre HORAYZON ; 85,7 % à moins de 1 km sur les panoramas acceptés (68) [V] | Idem |
| geo-sleuth | https://github.com/Oldcircle/geo-sleuth | MIT, ≈ 986 étoiles [V] | « Skill » d'agent ; `terrain.py` rend des skylines 360° depuis des tuiles Terrarium AWS et les compare | Référence MIT. Courbure non documentée, validation légère |

### 2.4 Matching et indexation

| Outil | Licence | Usage pour nous |
|---|---|---|
| **NumPy FFT** | BSD | **Corrélation circulaire normalisée** sur le cap, en un seul calcul par candidat. C'est le cœur de la baseline [D] |
| **FAISS** (`faiss-cpu`) | MIT [V] | Index de descripteurs **invariants au cap**, par exemple le module de la FFT du profil [D]. Pour 10 000 à 30 000 points, la force brute NumPy suffit ; FAISS devient utile au-delà (voir la section 6) |
| tslearn / dtaidistance | BSD-2 / Apache-2.0 [V] | DTW pour réordonner les candidats si la focale ou les crêtes proches déforment le profil (option) |
| dtw-python | GPL-3.0 [V] | À éviter (licence) |

---

## 3. Datasets publics avec vérité terrain

| Dataset | Lien | Contenu | Vérité terrain | Accès aujourd'hui | Intérêt pour nous |
|---|---|---|---|---|---|
| **GeoPose3K** | http://cphoto.fit.vutbr.cz/geoPose3K/ ; archive de 40 Go `merlin.fit.vutbr.cz/elevation/geoPose3K_final_publish.tar.gz` [V] | 3 111 photos Flickr, Alpes | Position + cap/tangage/roulis + champ de vue ; profondeur, normales, labels rendus [V] | **Téléchargeable** ; licence non précisée, usage recherche [M] | Évaluation à plus grande échelle ; **filtrer les photos de la zone Mont-Blanc** |
| **CH1 / CH2 (ETH)** | Original en 404. Copies Wayback [V] : `http://web.archive.org/web/2021id_/https://cvg.ethz.ch/research/mountain-localization/CH1.tgz` (236 Mo), idem `CH2.tgz` (1,2 Go) | 203 et 948 photos, Suisse | Position + orientation 3D ; champ de vue EXIF (imprécis, selon la page) | **Wayback seulement** | Comparaison avec l'ETH ; CH1 a des **skylines corrigées à la main** |
| **CrossLocate** | http://cphoto.fit.vutbr.cz/crosslocate/dataset/ [V] | 12 353 photos ; Sparse = 37 k rendus ; Uniform = 10,7 M rendus (profondeur, silhouette, segments) | Latitude, longitude, champ de vue, cap par image (`.npy`) | **Téléchargeable** (photos 15 Go) | **Valider notre moteur de rendu de silhouettes** contre le leur |
| **LandscapeAR — Alps Chamonix** | https://github.com/brejchajan/LandscapeAR#downloads [V] | 5,1 Go ; centre 45,997 N / 7,056 E, **rayon de 30 km : contient les Fiz** (≈ 26 km du centre) | Pose 6 degrés de liberté + intrinsèques (structure from motion alignée sur le MNT) | **Téléchargeable** (Flickr CC, non commercial) | ★ **Le meilleur jeu de test supplémentaire pour notre zone** |
| Alps100K | https://merlin.fit.vutbr.cz/elevation/ [V] | 98 k photos | GPS + altitude seulement, **pas d'orientation** | Formulaire | Non prioritaire |
| VENTURI Mountain | https://tev.fbk.eu/resources/venturimountain [V] | 12 vidéos de smartphone (Trentin) | Correspondances image↔MNT manuelles | Google Drive | Orientation seulement |
| SkyFinder | https://cs.valdosta.edu/~rpmihail/skyfinder/ (Zenodo 5884485) [V] | 90 k images, 53 webcams | **Masques ciel**, météo ; pas de pose | CC BY 4.0 | Évaluer ou fine-tuner le masque ciel |
| Multi-period Mountainous (Chen & Lian 2026) | https://ieee-dataport.org/documents/multi-period-mountainous-visual-geolocation-dataset-daytime-and-nighttime-scalable [V] | ≈ 700 km², jour et nuit | Cap par éphémérides | **Payant** (IEEE DataPort) | Non |

---

## 4. Sources MNT pour la France et pour les Fiz (téléchargement concret)

### 4.1 IGN RGE ALTI — Géoplateforme
- **Licence :** Licence Ouverte Etalab 2.0 [V].
- **Système de référence :** Lambert-93 (EPSG:2154) + altitudes NGF-IGN69 (EPSG:5720) [V].
- **Flux ATOM, filtrable par département** [V] : `https://data.geopf.fr/telechargement/resource/RGEALTI?zone=D074`
- **Haute-Savoie, édition 2020-10-15** [V, tailles obtenues par requête HEAD] :

  | Produit | Taille |
  |---|---|
  | **5 m, format ASC** | **329 Mo** (un seul fichier .7z) |
  | 1 m, format ASC | ≈ 5,0 Go (7z en deux volumes) |
  | 1 m, format TIF | ≈ 7,7 Go |
  | Savoie (D073), 5 m | 448 Mo, utile vers le sud |

  ```
  curl -O "https://data.geopf.fr/telechargement/download/RGEALTI/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D074_2020-10-15/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D074_2020-10-15.7z"
  ```
- **WMS en valeurs brutes (float32)** [V, testé] : `https://data.geopf.fr/wms-r/wms`
  - couche `ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES`, format `image/x-bil;bits=32` ou `image/geotiff` ;
  - 5010×5010 px au maximum par requête ;
  - **le WCS est en 404 : ne pas compter dessus.**
- **API d'altimétrie ponctuelle** [V] : `https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json?lon=6.72&lat=46.0&resource=ign_rge_alti_wld&zonly=true`. Utile pour contrôler l'altitude GPS des photos.
- **MNT LiDAR HD à 0,5 m** [V] :
  - WMS, couche `IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93` ;
  - **testé sur la zone des Fiz : 100 % de valeurs valides (1 662 à 2 427 m).**
- **BD ALTI 25 m :** obsolète, remplacée par RGE ALTI [V/D].

### 4.2 Copernicus DEM GLO-30 (AWS Open Data)
- **Accès :** bucket `s3://copernicus-dem-30m/`, sans identifiant (`--no-sign-request`), ou en HTTPS direct [V].
- **Nom de tuile :** `Copernicus_DSM_COG_10_N46_00_E006_00_DEM/Copernicus_DSM_COG_10_N46_00_E006_00_DEM.tif` (Cloud-Optimized GeoTIFF). Pour les Fiz et leur horizon : N45E006, N45E007, N46E006, N46E007 [V].
  ```
  gdal_translate /vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N46_00_E006_00_DEM/Copernicus_DSM_COG_10_N46_00_E006_00_DEM.tif n46e006.tif
  ```
- **Caractéristiques [M, spécification standard] :**
  - WGS84 + altitudes EGM2008 ;
  - **modèle de surface (DSM)** : inclut forêt et bâti ;
  - données TanDEM-X 2011-2015.
- **Licence :** Copernicus, gratuite [V].
- **Outils :** `rasterio` ; `dem_stitcher` (`stitch_dem(bbox, 'glo_30')`) [V].

### 4.3 Hors de France : indispensable pour l'horizon des Fiz
- **swissALTIRegio, 10 m (swisstopo)** ★ [V] :
  - **un seul GeoTIFF de 10,1 Go**, emprise de 4,52 à 11,99 E et de 44,84 à 48,79 N ;
  - **c'est un MNT (terrain nu) qui couvre aussi la France et l'Italie voisines** : Fiz, Mont-Blanc côté italien, Vallée d'Aoste et Valais dans un seul système de coordonnées (LV95 / LN02) ;
  - URL : `https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio/swissaltiregio_2056_5728.tif` ;
  - licence OGD swisstopo, citation de la source obligatoire.
  - À vérifier en phase 2 [D] : si le fichier est un COG, un découpage via `/vsicurl/` évite de télécharger les 10 Go.
- **swissALTI3D (0,5 / 2 m) :** API STAC `https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items?bbox=...`, tuiles de 1 km, Suisse seulement [V].
- **TINITALY 1.1 (INGV, 10 m), CC BY 4.0** [V] : https://tinitaly.pi.ingv.it/Download_Area1_1.html
- **Géoportail de la Vallée d'Aoste** [V] : MNT LiDAR 2005/2008 à 2, 10 et 25 m, CC BY. Téléchargement interactif seulement.
- **viewfinderpanoramas 1″** [V] : MNT corrigé à la main pour les Alpes, exemple `https://viewfinderpanoramas.org/dem1/N46E006.zip`. **Le MNT utilisé par Brejcha et CrossLocate** : cohérent pour comparer.
- **EU-DEM :** retiré, renvoie vers Copernicus [V].

### 4.4 Assemblage et datums
- **Écarts verticaux** entre NGF-IGN69, LN02, EGM2008 et le datum italien [M/D] :
  - de l'ordre de 0,5 à 1 m au plus ;
  - 1 m à 5 km représente 0,011° : **négligeable** devant l'erreur des MNT et la taille d'un pixel (0,015 à 0,05°) ;
  - pas de correction nécessaire pour le POC.
- **GLO-30 au-delà de 5 km suffit-il ?**
  - L'erreur angulaire vaut environ Δh/d : 10 m d'erreur donnent 0,11° à 5 km et 0,03° à 20 km [calculé].
  - **Oui pour les crêtes lointaines rocheuses ou glaciaires.**
  - **Non pour les crêtes forestières proches** : biais de canopée de 10 à 30 m, parce que c'est un DSM.
  - swissALTIRegio est un MNT de 10 m avec la même couverture : **meilleur que GLO-30 ici** [D].

---

## 5. Apport possible des modèles récents

| Brique | Modèle (licence) | Chiffres | Verdict pour le POC [D] |
|---|---|---|---|
| Masque ciel | SegFormer ou Mask2Former ADE20K, plus SAM2 pour le bord (Apache) | Ciel ≈ 92 % IoU (modèle tiers) ; SkyFinder **neige 51 mIoU** contre ciel clair 73 (La Place 2017) [V] | **Oui, dès la baseline**, avec une **skyline corrigée à la main (oracle)** pour séparer les erreurs de segmentation des erreurs de matching |
| Profondeur monoculaire → crêtes internes | **MoGe-2/3 (MIT)** : profondeur + champ de vue + masque ciel [V] ; **Depth Anything 3 Mono-Large (Apache)** [V] ; Depth Pro (licence Apple, bords les plus nets [V]) | **Aucun benchmark à l'échelle du km.** Les modèles métriques de Depth Anything V2 sont plafonnés à 80 m [V] ; Depth Pro échoue sur la diffusion volumique (brume) [V]. Seul ViUT (Čadík 2023) vise des dizaines de km [V] | **Après la baseline.** L'ordre relatif des profondeurs suffit à séparer les plans de crête (contours d'occultation). Aucun papier ne le fait encore [V-négatif]. CrossLocate montre que la profondeur rendue est la meilleure modalité, ce qui confirme l'intérêt |
| Calibration (roulis, tangage, champ de vue) | **GeoCalib** (code Apache, poids CC-BY) [V] | Médiane : roulis 0,3 à 0,4°, tangage 0,9 à 1,9°, **champ de vue 3 à 5°** [V] | **Utile** comme a priori de roulis et tangage. Garder la focale libre dans le matcher (EXIF ± balayage) |
| Pré-filtre global | GeoCLIP (MIT, pip), G3 (Apache), OSV-5M (MIT) ; PIGEON **sans poids publiés** [V] | Im2GPS3k à moins de 25 km : 34 à 45 % (toutes scènes) ; **scènes naturelles : 17,5 % à moins de 25 km** [V] | **Inutile ici** : la zone est connue. À ne considérer que pour une généralisation à « n'importe où dans les Alpes », comme a priori souple à 200-750 km |
| Matching dense photo↔rendu | RoMa, LoFTR, LightGlue, MASt3R | **Aucun benchmark photo↔rendu MNT** ; échec de SuperGlue dans Wu 2024 [V] | Raffinement final éventuel, avec des rendus texturés par orthophoto. Hors POC |

---

## 6. Recommandation d'architecture pour le POC

Le principe est celui de la phase 2 que tu as décrite. Les ajustements ci-dessous découlent de la littérature et de la géographie des Fiz [D].

### 6.1 Données MNT : deux emprises distinctes
- **Zone des points de vue** = emprise GPS + 15 km, grille de 250 m.
  - Pour une emprise d'environ 30×30 km : environ 120×120 ≈ **15 000 points** (à recalculer avec les vraies positions).
- **Zone de rendu** = zone des points de vue + **100 km**, pour voir le Valais et le Grand Combin.
- **MNT empilé, réprojeté en UTM 32N (EPSG:32632) :**
  1. **RGE ALTI 5 m** (Haute-Savoie, plus Savoie si nécessaire) sur la zone des points de vue + 5 km ;
  2. **swissALTIRegio 10 m** (découpé) partout ailleurs ;
  3. **GLO-30** pour combler les trous. Plan de repli si swissALTIRegio pose problème.
- **Résolution de rendu variable avec la distance** (5 m près, 10 à 30 m loin). Ce n'est pas de la sur-ingénierie : c'est ce qui rend le calcul faisable.

### 6.2 Base de profils d'horizon
- **Calcul :** pour chaque point de vue, **lancer de rayons radial maison en NumPy + Numba**.
  - 1 440 azimuts (0,25°), pas croissant avec la distance, jusqu'à 100 km.
  - Correction de courbure et de réfraction Δh = d²/(2R)·(1−k), k = 0,13.
  - Caméra à 1,7 m au-dessus du sol.
- **Pourquoi pas HORAYZON :** sa compilation Windows est pénible. Le code maison reste court et lisible, et sera **validé contre GRASS r.horizon ou HORAYZON** sur 3 à 5 points.
- **Stocker par point :**
  - le profil d'élévation (1 440 float16) ;
  - **la distance de l'occultant par azimut** (indispensable pour l'analyse des échecs « crête proche ») ;
  - l'altitude du sol.
- **Index :**
  - `profiles.npy` + `points.parquet` ;
  - un index FAISS sur un descripteur invariant au cap (module de la FFT du profil), pour respecter la consigne ;
  - mais vu la taille (≈ 10⁴ points), **le matching exhaustif est faisable et sert de référence** : on mesure ce que l'index perd en rappel.

### 6.3 Extraction de l'horizon photo
1. **Masque ciel :** SegFormer ou Mask2Former ADE20K (classe ciel), en local.
2. **Skyline :** pixel non-ciel le plus haut par colonne, lissé par programmation dynamique (à la Ahmad).
3. **Colonnes douteuses :** masquer celles où le ciel est incertain (nuages, neige) au lieu de les forcer.
4. **Conversion en profil angulaire** avec la focale EXIF (focale équivalente 35 mm → champ de vue) :

       azimut_relatif = atan((x − cx) / f)
       élévation      = atan((cy − y) / f)

5. **Skyline oracle :** tracer à la main (outil simple) la skyline des 9 photos. **C'est l'expérience la plus informative du POC** : elle sépare l'erreur de segmentation de l'erreur de matching (facteur 2 observé dans la littérature).

### 6.4 Matching baseline (sans apprentissage)
- **Balayage :**
  - cap 0 à 360° par **corrélation circulaire FFT** (gratuite) ;
  - focale ± 10 % autour de l'EXIF (ou balayage de 20 à 100 mm équivalent si absente) ;
  - **tangage inconnu** : absorbé en retirant la moyenne du profil ;
  - **roulis** : balayage de ± 3° ou estimation GeoCalib.
- **Score :** NCC sur les colonnes valides. Fusion des scores par point (maximum sur cap × focale × roulis).
- **Raffinement local** : grille de 25 m dans un rayon de 500 m autour des 20 meilleurs candidats. Indispensable aux Fiz, où l'horizon proche varie fortement en 250 m.

### 6.5 Évaluation
- **Mesures :** rang de la vraie position (point de grille le plus proche du GPS), top-1/5/20, erreur en mètres, erreur de cap.
  - Le cap de référence vient de `GPSImgDirection` si l'EXIF en contient (iPhone), sinon du cap estimé au point GPS vrai.
- **Visualisations :** profil photo superposé au profil MNT du meilleur candidat et du vrai point, plus la carte des scores.
- **Taille de l'échantillon :** 9 photos, c'est très peu pour des statistiques. Je propose d'ajouter **le sous-ensemble LandscapeAR « Alps Chamonix » situé dans la zone**, et éventuellement les photos GeoPose3K de la zone. Les chiffres ont ainsi un sens, et les photos LandscapeAR ont une pose 6 degrés de liberté.

### 6.6 Ce que j'attends (hypothèses à tester, pas des promesses) [D]
- **Horizon oracle + focale EXIF :** bons résultats probables sur les photos avec un horizon lointain dégagé (vers le Mont-Blanc), sur une zone d'environ 1 000 à 2 000 km².
- **Horizon automatique :** dégradation sur les photos avec neige, nuages ou brume.
- **Échecs attendus :** photos où l'horizon est une paroi proche (falaises des Fiz) avec une grille de 250 m. Le raffinement local devrait en récupérer une partie.
- **Amélioration n°1 la plus probable après la baseline :** les **crêtes internes**, via des contours d'occultation tirés d'une profondeur monoculaire et confrontés aux discontinuités de profondeur du rendu MNT. C'est la direction validée indirectement par CrossLocate.

### 6.7 Stack logicielle (Python, environnement `uv`)
- **Base :** `numpy`, `numba`, `rasterio`, `pyproj`, `pandas` / `pyarrow`, `exifread` ou `Pillow`, `matplotlib`, `faiss-cpu`.
- **Segmentation :** `torch` + `transformers` (SegFormer/Mask2Former), optionnellement `sam2`.
- **Hors ligne :** **aucune photo n'est envoyée en ligne.** Seuls les MNT et les poids de modèles sont téléchargés.

---

## 7. Limites de ce document
- La recherche a été faite par quatre agents en parallèle, avec consultation des pages et PDF. J'ai revérifié moi-même les dépôts douteux (SkylineGeolocation, horizon-locate, geo-sleuth, HORAYZON, CrossLocate), l'arXiv de HorizonNet et le fichier swissALTIRegio.
- **Non vérifiés :**
  - les chiffres ETH originaux (88 % sur CH1) ;
  - les licences des datasets GeoPose3K, CrossLocate et CH1 ;
  - les chiffres détaillés de CMLocate et de Kubička 2026 ;
  - les valeurs exactes des écarts de datums.
- Les tableaux d'état de l'art de la thèse Brejcha ont une mise en page cassée dans le PDF : chiffres approximatifs.
