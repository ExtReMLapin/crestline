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

from . import kernels

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


def query_words(P, M, soft=True):
    """Mots de la photo (et leurs variantes proches d'un seuil) : liste de (mot, direction du mot dans la photo)."""
    queries = []
    for scale in range(len(WIDTHS)):
        w, d, ok, rel = words_from_profile(P, M, circular=False, with_rel=True, scale=scale)
        queries += [(v, dq) for word, dq, r in zip(w[ok].tolist(), d[ok].tolist(), rel[ok])
                    for v in (soft_variants(word, r, scale) if soft else [word])]
    return queries


def _vote(votes, pts, dirs, dq, weight, k_center):
    """Un vote par entrée : cap de l'axe optique = direction du mot dans la base - décalage du mot dans la photo."""
    n_head = int(round(360 / HEADING_BIN))
    head = (dirs * STRIDE - (dq - k_center * BIN)) % 360
    hb = (head / HEADING_BIN).long() % n_head
    votes.index_add_(0, pts * n_head + hb, weight)


def _smooth_max(v):
    """Tolérance d'un cran sur le cap (somme glissante sur 3 cases), puis meilleur cap de chaque lieu."""
    return (v + v.roll(1, 1) + v.roll(-1, 1)).max(1)


_fused = {}


def smooth_max(v):
    """_smooth_max fusionné en un seul passage sur GPU (torch.compile) ; version simple sinon, ou en cas d'échec."""
    if v.is_cuda and "f" not in _fused:
        try:
            _fused["f"] = torch.compile(_smooth_max, dynamic=False)
            _fused["f"](v[:2])
        except Exception:                                   # pas de compilateur : version simple
            _fused["f"] = _smooth_max
    return (_fused.get("f", _smooth_max) if v.is_cuda else _smooth_max)(v)


def top_votes(votes, n_points, k, per_place=False):
    """k meilleurs (lieu, cap) après la tolérance d'un cran sur le cap ; per_place : k lieux distincts, chacun
    avec son meilleur cap (même liste de lieux, sans trier les 600 M de couples (lieu, cap))."""
    n_head = int(round(360 / HEADING_BIN))
    v = votes.view(n_points, n_head)
    if per_place:
        best, head = smooth_max(v)
        top = torch.topk(best, min(k, n_points))
        return top.indices.cpu().numpy(), (head[top.indices] * HEADING_BIN).cpu().numpy(), top.values.cpu().numpy()
    top = torch.topk((v + v.roll(1, 1) + v.roll(-1, 1)).view(-1), min(k, v.numel()))
    idx = top.indices.cpu().numpy()
    return idx // n_head, (idx % n_head) * HEADING_BIN, top.values.cpu().numpy()


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

    def query(self, P, M, k_center, k=1000, max_postings=2_000_000, device="cuda", soft=True, chunk=20_000_000):
        """P, M : profil photo (une hypothèse de tangage, cases de BIN degrés, case 0 = bord gauche) ;
        k_center : case de l'axe optique. Renvoie les k meilleurs (point, cap en degrés) et leurs votes."""
        n_head = int(round(360 / HEADING_BIN))
        votes = torch.zeros(self.n_points * n_head, dtype=torch.float32, device=device)
        spans = [(self.offsets[w], self.offsets[w + 1], w, dq) for w, dq in query_words(P, M, soft)]
        spans = [sp for sp in spans if 0 < sp[1] - sp[0] <= max_postings]       # absent, ou mot trop banal
        batch, size = [], 0                     # par paquets : un transfert et un index_add par paquet
        for i, sp in enumerate(spans):
            batch.append(sp)
            size += sp[1] - sp[0]
            if size >= chunk or i == len(spans) - 1:
                pts = np.concatenate([self.point[a:b] for a, b, _, _ in batch])
                dirs = np.concatenate([self.direction[a:b] for a, b, _, _ in batch])
                lens = torch.as_tensor([b - a for a, b, _, _ in batch], device=device)
                dq = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=device)
                w = torch.as_tensor(self.idf[[x[2] for x in batch]], device=device)
                _vote(votes, torch.as_tensor(pts, device=device).long(), torch.as_tensor(dirs, device=device).float(),
                      torch.repeat_interleave(dq, lens), torch.repeat_interleave(w, lens), k_center)
                batch, size = [], 0
        return top_votes(votes, self.n_points, k)


class GpuIndex:
    """Le même index entièrement sur GPU, sans perte par rapport à RidgeIndex.query :
    - seuls les mots utilisables sont gardés : un mot présent dans plus de max_postings lieux n'est jamais lu
      (horizons plats, pentes régulières), et c'est ~64 % des entrées sur la base suisse ;
    - lieu et direction dans un seul int32 (lieu << 9 | direction en degrés).
    Base suisse : 3,16 G entrées (18 Go sur disque) -> 1,15 G sur GPU (4,6 Go)."""

    DIR_BITS = 9

    def __init__(self, index, max_postings=2_000_000, device="cuda", chunk=50_000_000):
        assert index.n_points < 2 ** (31 - self.DIR_BITS)
        counts = np.diff(index.offsets)
        keep = counts <= max_postings
        self.offsets = torch.as_tensor(np.concatenate([[0], np.cumsum(np.where(keep, counts, 0))]), device=device)
        self.idf = torch.as_tensor(index.idf, device=device)
        self.n_points, self.device = index.n_points, device
        self.packed = torch.empty(int(self.offsets[-1]), dtype=torch.int32, device=device)
        dropped = np.nonzero(~keep & (counts > 0))[0]          # entrées gardées : entre deux mots écartés
        bounds = [0] + [int(x) for w in dropped for x in (index.offsets[w], index.offsets[w + 1])] + [int(index.offsets[-1])]
        out = 0
        for a, b in zip(bounds[0::2], bounds[1::2]):
            for i in range(a, b, chunk):
                j = min(i + chunk, b)
                pt = torch.from_numpy(np.array(index.point[i:j])).to(device)          # copie : mmap en lecture seule
                dr = torch.from_numpy(np.asarray(index.direction[i:j]).astype(np.int32)).to(device)
                self.packed[out:out + j - i] = (pt << self.DIR_BITS) | dr
                out += j - i

    @staticmethod
    def bytes_needed(index, max_postings=2_000_000):
        counts = np.diff(index.offsets)
        return int(counts[counts <= max_postings].sum()) * 4

    def default_tile(self):
        """Lieux par tuile de votes : ~1/4 du cache L2 (RTX 5090 : 96 Mo -> ~16 000 lieux), au plus 256 tuiles."""
        l2 = torch.cuda.get_device_properties(torch.device(self.device)).L2_cache_size
        return max(l2 // 4 // (int(round(360 / HEADING_BIN)) * 4), (self.n_points + 255) // 256)

    def query(self, queries, k_center, k=1000, chunk=100_000_000, tile=None):
        """queries : query_words(...). Renvoie k lieux distincts (leur meilleur cap) et leurs votes.
        tile : lieux par tuile de votes (0 : sans tuiles ; défaut : selon le cache L2)."""
        dev = self.device
        n_head = int(round(360 / HEADING_BIN))
        size = self.n_points * n_head
        shared = getattr(self, "votes", None)                 # tableau partagé entre morceaux (ShardedIndex)
        votes = shared[:size].zero_() if shared is not None and len(shared) >= size else \
            torch.zeros(size, dtype=torch.float32, device=dev)
        if not queries:
            return top_votes(votes, self.n_points, k, per_place=True)
        words = torch.tensor([w for w, _ in queries], dtype=torch.long, device=dev)
        dq = torch.tensor([d for _, d in queries], dtype=torch.float32, device=dev)
        a, n = self.offsets[words], self.offsets[words + 1] - self.offsets[words]
        live = n > 0
        a, n, dq, idf = a[live], n[live], dq[live], self.idf[words[live]]
        if len(n) == 0:                                       # aucun mot présent dans ce morceau
            return top_votes(votes, self.n_points, k, per_place=True)
        if kernels.available(dev) and STRIDE == 1.0 and HEADING_BIN == 1.0:
            tile = self.default_tile() if tile is None else tile
            if tile:
                a, n, dq, idf = self._tiles(a, n, dq, idf, tile)
            ends = torch.cumsum(n, 0)
            kernels.vote(self.packed, ends, a - (ends - n), dq, idf, votes, int(ends[-1]), k_center * BIN,
                         n_head, self.DIR_BITS)
            return top_votes(votes, self.n_points, k, per_place=True)
        ends = torch.cumsum(n, 0)
        lo = 0
        while lo < len(n):                                    # paquets d'au plus `chunk` entrées
            hi = max(int(torch.searchsorted(ends, ends[lo] - n[lo] + chunk, right=True)), lo + 1)
            nn = n[lo:hi]
            rep = torch.repeat_interleave(torch.arange(hi - lo, device=dev), nn)
            start = torch.cumsum(nn, 0) - nn
            e = self.packed[a[lo:hi][rep] + (torch.arange(len(rep), device=dev) - start[rep])]
            _vote(votes, (e >> self.DIR_BITS).long(), (e & ((1 << self.DIR_BITS) - 1)).float(),
                  dq[lo:hi][rep], idf[lo:hi][rep], k_center)
            lo = hi
        return top_votes(votes, self.n_points, k, per_place=True)

    def _tiles(self, a, n, dq, idf, tile):
        """Découpe chaque mot en segments par tuile de `tile` lieux, rangés tuile par tuile : les votes d'une
        tuile restent dans le cache L2. Dans un mot, les entrées sont rangées par lieu (dichotomie)."""
        dev = self.device
        n_tiles = (self.n_points + tile - 1) // tile
        bounds = torch.arange(n_tiles + 1, device=dev, dtype=torch.int64) * tile << self.DIR_BITS
        lo = a[:, None].expand(-1, n_tiles + 1).clone()
        hi = (a + n)[:, None].expand(-1, n_tiles + 1).clone()
        for _ in range(int(n.max()).bit_length() + 1):        # premier rang dont le lieu atteint la limite
            mid = (lo + hi) // 2
            below = (self.packed[mid.clamp(max=len(self.packed) - 1)].long() < bounds[None]) & (lo < hi)
            lo = torch.where(below, mid + 1, lo)
            hi = torch.where(below | (lo >= hi), hi, mid)
        seg_a = lo[:, :-1].T.reshape(-1)
        seg_n = (lo[:, 1:] - lo[:, :-1]).T.reshape(-1)
        slot = torch.arange(len(n), device=dev).repeat(n_tiles)
        keep = seg_n > 0
        return seg_a[keep], seg_n[keep], dq[slot[keep]], idf[slot[keep]]


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

    def gpu_bytes(self):
        return sum(GpuIndex.bytes_needed(s) for s in self.shards)

    def to_gpu(self, device="cuda", reserve=0.0):
        """Copie les morceaux sur GPU (GpuIndex) un par un, tant qu'il reste `reserve` octets libres après ;
        les autres restent en mémoire centrale. Mêmes lieux proposés. Renvoie le nombre de morceaux sur GPU."""
        self.gpu = []
        for s in self.shards:
            fits = torch.cuda.mem_get_info(torch.device(device))[0] - reserve > GpuIndex.bytes_needed(s)
            self.gpu.append(GpuIndex(s, device=device) if fits else None)
        on_gpu = [g for g in self.gpu if g is not None]
        if on_gpu:                                            # un seul tableau de votes pour tous les morceaux
            votes = torch.empty(max(g.n_points for g in on_gpu) * int(round(360 / HEADING_BIN)), device=device)
            for g in on_gpu:
                g.votes = votes
        return len(on_gpu)

    def query(self, P, M, k_center, n_places=1000, device="cuda"):
        """Renvoie jusqu'à n_places lieux distincts (indices globaux), du plus voté au moins voté."""
        pts, votes = [], []
        gpu = getattr(self, "gpu", None) or [None] * len(self.shards)
        queries = query_words(P, M) if any(g is not None for g in gpu) else None
        for i, (start, s) in enumerate(zip(self.starts, self.shards)):
            if gpu[i] is not None:
                p, _, v = gpu[i].query(queries, k_center, k=n_places)       # déjà des lieux distincts
            else:
                p, _, v = s.query(P, M, k_center, k=20 * n_places, device=device)
            pts.append(p + start)
            votes.append(v)
        pts, votes = np.concatenate(pts), np.concatenate(votes)
        order = np.argsort(-votes, kind="stable")
        _, first = np.unique(pts[order], return_index=True)
        return pts[order][np.sort(first)][:n_places]
