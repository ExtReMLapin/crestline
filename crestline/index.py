"""Index de « mots de crête » : présélection rapide des lieux candidats (dans l'esprit de Baatz et al. 2012).

Construction (une fois) : chaque panorama est découpé en petits morceaux de crête (10° et 4° de large),
chacun devient un « mot » (sa forme, quantifiée), rangé dans un index inversé.

    morceau de 10°      8 valeurs relatives au centre      quantifiées (5 niveaux)
        ╱╲               +1.5 +0.8 +0.1  0 -0.3 ...          4  3  2  2  1 ...  ──▶  mot n° 48 213

    mot 48 213 ──▶ [(lieu 1 203 554, direction 127°), (lieu 88 120, direction 15°), ...]

Requête : chaque mot de la photo vote pour (lieu, cap = direction stockée - position du mot dans la photo).
Un vrai lieu accumule des votes sur UN MÊME cap ; les mots banals (horizons plats) sont ignorés, les autres
pondérés par leur rareté (idf).

    lieu A, cap 161° : ████████████████  ← candidat
    lieu B, cap  20° : ███
"""
from pathlib import Path

import numpy as np
import torch

BIN = 0.25                    # degrés par case des profils de la base
WIDTHS = (10.0, 4.0)          # largeurs des mots (degrés) : deux échelles, pour les photos au champ étroit
N_SAMPLES = 8                 # valeurs par mot
STRIDE = 1.0                  # pas entre deux mots (degrés)
THRESHOLDS = (-1.2, -0.4, 0.4, 1.2)   # degrés, relatif au centre, pour un mot de 10° (proportionnel à la largeur)
LEVELS = len(THRESHOLDS) + 1
VOCAB = LEVELS ** N_SAMPLES         # mots par échelle ; mot global = échelle * VOCAB + mot
HEADING_BIN = 1.0             # degrés, résolution du vote sur le cap
MIN_RANGE = 0.3               # degrés : un morceau plus plat que ça ne dit rien (mot ignoré)
SOFT_MARGIN = 0.15            # degrés : valeur de la photo aussi proche d'un seuil -> on vote aussi pour le mot voisin


def thresholds(width):
    return torch.tensor(THRESHOLDS) * width / 10.0


def words_from_profile(prof, valid=None, start=0.0, step=STRIDE, circular=True, with_rel=False, scale=0):
    """prof : (..., A) élévations sur des cases de BIN degrés. Renvoie (mots (..., K) int64, directions (K,)
    en degrés du centre de chaque mot, masque (..., K) des mots utilisables [, valeurs relatives (..., K, N)])."""
    prof = torch.as_tensor(prof, dtype=torch.float32)
    A = prof.shape[-1]
    width = WIDTHS[scale]
    span = width / BIN
    offsets = (torch.arange(N_SAMPLES) - (N_SAMPLES - 1) / 2) * span / (N_SAMPLES - 1)     # en cases
    if circular:
        centers = torch.arange(0, A, step / BIN)
    else:
        centers = torch.arange(start / BIN + span / 2, A - span / 2 - 1e-6, step / BIN)
    pos = centers[:, None] + offsets[None]                                              # (K, N)
    if circular:
        pos = pos % A
    lo = pos.floor().long()
    w = pos - lo
    hi = (lo + 1) % A if circular else (lo + 1).clamp(max=A - 1)
    vals = prof[..., lo] * (1 - w) + prof[..., hi] * w                                  # (..., K, N)
    c_lo = centers.floor().long() % A
    c_w = centers - centers.floor()
    center = prof[..., c_lo] * (1 - c_w) + prof[..., (c_lo + 1) % A] * c_w
    rel = vals - center[..., None]
    q = torch.bucketize(rel, thresholds(width))
    word = (q * (LEVELS ** torch.arange(N_SAMPLES))).sum(-1) + scale * VOCAB
    ok = (vals.amax(-1) - vals.amin(-1)) >= MIN_RANGE * width / 10.0
    if valid is not None:
        v = torch.as_tensor(valid, dtype=torch.bool)
        ok &= v[..., lo].all(-1) & v[..., hi].all(-1)
    if with_rel:
        return word, centers * BIN, ok, rel
    return word, centers * BIN, ok


def soft_variants(word, rel, scale, n_flip=2, margin=SOFT_MARGIN):
    """Variantes d'un mot de requête : les n_flip valeurs les plus proches d'un seuil (à moins de margin
    degrés) basculent dans le niveau voisin. Renvoie la liste des mots (le mot exact en premier)."""
    thr = thresholds(WIDTHS[scale])
    margin = margin * WIDTHS[scale] / 10.0
    dist = (rel[:, None] - thr[None]).abs()                     # (N, T)
    d, t = dist.min(1)
    q = torch.bucketize(rel, thr)
    order = torch.argsort(d)[:n_flip]
    flips = [(int(i), int(t[i])) for i in order if d[i] < margin]
    out = [int(word)]
    for i, ti in flips:
        alt = ti if q[i] == ti + 1 else ti + 1                  # niveau de l'autre côté du seuil
        delta = (alt - int(q[i])) * LEVELS ** i
        out += [w + delta for w in out]
    return out


class RidgeIndex:
    """Index inversé en mémoire : postings triés par mot (CSR)."""

    def __init__(self, offsets, point, direction, n_points):
        self.offsets, self.point, self.direction, self.n_points = offsets, point, direction, n_points
        counts = np.diff(offsets)
        self.idf = np.log((counts.sum() + 1) / (counts + 1)).astype(np.float32)

    @classmethod
    def build(cls, elev, batch=20000):
        """elev : (N, A) profils de la base (mmap accepté)."""
        words, points, dirs = [], [], []
        for i in range(0, len(elev), batch):
            e = torch.as_tensor(np.asarray(elev[i:i + batch], np.float32)).clamp(min=-15)
            for scale in range(len(WIDTHS)):
                w, d, ok = words_from_profile(e, scale=scale)
                p = torch.arange(i, i + len(e))[:, None].expand_as(w)
                dd = (d / STRIDE).round().to(torch.int16)[None].expand_as(w)
                words.append(w[ok].to(torch.int32).numpy())
                points.append(p[ok].to(torch.int32).numpy())
                dirs.append(dd[ok].numpy())
        words, points, dirs = np.concatenate(words), np.concatenate(points), np.concatenate(dirs)
        order = np.argsort(words, kind="stable")
        offsets = np.zeros(len(WIDTHS) * VOCAB + 1, np.int64)
        np.add.at(offsets, words + 1, 1)
        return cls(np.cumsum(offsets), points[order], dirs[order], len(elev))

    def save(self, d):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "offsets.npy", self.offsets)
        np.save(d / "point.npy", self.point)
        np.save(d / "direction.npy", self.direction)
        np.save(d / "n_points.npy", np.array(self.n_points))

    @classmethod
    def load(cls, d):
        d = Path(d)
        return cls(np.load(d / "offsets.npy"), np.load(d / "point.npy", mmap_mode="r"),
                   np.load(d / "direction.npy", mmap_mode="r"), int(np.load(d / "n_points.npy")))

    def query(self, P, M, k_center, k=1000, max_postings=2_000_000, device="cuda", soft=True):
        """P, M : profil photo (une hypothèse de tangage, cases de BIN degrés, case 0 = bord gauche) ;
        k_center : case de l'axe optique. Renvoie les k meilleurs (point, cap en degrés) et leurs votes."""
        n_head = int(round(360 / HEADING_BIN))
        votes = torch.zeros(self.n_points * n_head, dtype=torch.float32, device=device)
        queries = []
        for scale in range(len(WIDTHS)):
            w, d, ok, rel = words_from_profile(P, M, circular=False, with_rel=True, scale=scale)
            queries += [(v, dq) for word, dq, r in zip(w[ok].tolist(), d[ok].tolist(), rel[ok])
                        for v in (soft_variants(word, r, scale) if soft else [word])]
        for word, dq in queries:
            a, b = self.offsets[word], self.offsets[word + 1]
            if b - a == 0 or b - a > max_postings:            # absent, ou mot trop banal
                continue
            pts = torch.from_numpy(np.array(self.point[a:b])).to(device).long()
            dirs = torch.from_numpy(np.array(self.direction[a:b])).to(device).float() * STRIDE
            # cap de l'axe optique = direction du mot dans la base - décalage du mot dans la photo
            head = (dirs - (dq - k_center * BIN)) % 360
            hb = (head / HEADING_BIN).long() % n_head
            votes.index_add_(0, pts * n_head + hb, torch.full_like(head, float(self.idf[word])))
        # tolérance d'un cran sur le cap : somme glissante sur 3 cases
        v = votes.view(self.n_points, n_head)
        v = v + v.roll(1, 1) + v.roll(-1, 1)
        top = torch.topk(v.view(-1), k)
        idx = top.indices.cpu().numpy()
        return idx // n_head, (idx % n_head) * HEADING_BIN, top.values.cpu().numpy()


class ShardedIndex:
    """Plusieurs index (un par morceau de base), interrogés l'un après l'autre et fusionnés.
    C'est aussi le moyen de passer à l'échelle : un morceau par région, ou par machine.
    L'idf est commun à tous les morceaux, pour que les votes restent comparables."""

    def __init__(self, index_dirs):
        self.shards = [RidgeIndex.load(d) for d in index_dirs]
        counts = sum(np.diff(s.offsets) for s in self.shards)
        idf = np.log((counts.sum() + 1) / (counts + 1)).astype(np.float32)
        for s in self.shards:
            s.idf = idf
        self.starts = np.cumsum([0] + [s.n_points for s in self.shards])

    def query(self, P, M, k_center, n_places=1000, device="cuda"):
        """Renvoie jusqu'à n_places lieux distincts (indices globaux), du plus voté au moins voté."""
        pts, votes = [], []
        for start, s in zip(self.starts, self.shards):
            p, _, v = s.query(P, M, k_center, k=20 * n_places, device=device)
            pts.append(p + start)
            votes.append(v)
        pts, votes = np.concatenate(pts), np.concatenate(votes)
        order = np.argsort(-votes, kind="stable")
        _, first = np.unique(pts[order], return_index=True)
        return pts[order][np.sort(first)][:n_places]
