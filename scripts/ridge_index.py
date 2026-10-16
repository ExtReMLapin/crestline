"""Index de « mots de crête » : présélection rapide des lieux candidats, indépendante de la taille de la base.

Principe (dans l'esprit de Baatz et al. 2012) :
- un mot = forme d'un petit morceau d'horizon : fenêtre de WIDTHS degrés (deux échelles), N_SAMPLES valeurs d'élévation
  relatives au centre de la fenêtre (insensible au tangage), chaque valeur quantifiée en LEVELS niveaux ;
- construction (hors ligne) : pour chaque point de la base et chaque direction (pas STRIDE), le mot est
  rangé dans un index inversé mot -> (point, direction) ;
- requête : les mots de la photo votent pour (point, cap = direction stockée - direction dans la photo).
  Un vrai lieu accumule des votes sur un même cap. Les mots trop fréquents (horizons plats) sont ignorés,
  les autres sont pondérés par leur rareté (idf).
"""
import math
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


def query_words(P, M, soft=True):
    """Mots de la photo (avec leurs variantes proches d'un seuil) : liste de (mot, direction dans la photo)."""
    queries = []
    for scale in range(len(WIDTHS)):
        w, d, ok, rel = words_from_profile(P, M, circular=False, with_rel=True, scale=scale)
        queries += [(v, dq) for word, dq, r in zip(w[ok].tolist(), d[ok].tolist(), rel[ok])
                    for v in (soft_variants(word, r, scale) if soft else [word])]
    return queries


def _smooth_max(v):
    """Meilleur cap de chaque lieu après somme glissante sur 3 cases (circulaire) : (max, cap)."""
    s = v + v.roll(1, 1) + v.roll(-1, 1)
    return s.max(1)


_smooth_max_fused = torch.compile(_smooth_max, dynamic=False)   # un seul passage sur le tableau des votes


def top_votes(votes, n_points, k, per_place=False):
    """Tolérance d'un cran sur le cap (somme glissante sur 3 cases, sur place) puis k meilleurs (lieu, cap) ;
    per_place : un seul couple par lieu (son meilleur cap), soit k lieux distincts."""
    n_head = int(round(360 / HEADING_BIN))
    v = votes.view(n_points, n_head)
    if per_place:                                       # meilleur cap de chaque lieu, puis k meilleurs lieux
        best, head = _smooth_max_fused(v)
        top = torch.topk(best, min(k, n_points))
        idx = top.indices
        return idx.cpu().numpy(), (head[idx] * HEADING_BIN).cpu().numpy(), top.values.cpu().numpy()
    first, last = v[:, :1].clone(), v[:, -1:].clone()
    left = torch.cat([last, v[:, :-1]], 1)            # v.roll(1) ; une seule copie temporaire
    left += v
    left[:, :-1] += v[:, 1:]
    left[:, -1:] += first
    top = torch.topk(left.view(-1), k)
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

    def _vote(self, votes, spans, k_center, n_head, device):
        lens = torch.as_tensor(np.array([b - a for a, b, _, _ in spans]), device=device)
        pts = torch.as_tensor(np.concatenate([self.point[a:b] for a, b, _, _ in spans]), device=device).long()
        dirs = torch.as_tensor(np.concatenate([self.direction[a:b] for a, b, _, _ in spans]), device=device)
        dq = torch.repeat_interleave(torch.tensor([x[3] for x in spans], device=device, dtype=torch.float32), lens)
        w = torch.repeat_interleave(torch.as_tensor(self.idf[[x[2] for x in spans]], device=device), lens)
        # cap de l'axe optique = direction du mot dans la base - décalage du mot dans la photo
        head = (dirs.float() * STRIDE - (dq - k_center * BIN)) % 360
        hb = (head / HEADING_BIN).long() % n_head
        votes.index_add_(0, pts * n_head + hb, w)

    def query(self, P, M, k_center, k=1000, max_postings=2_000_000, device="cuda", soft=True, chunk=20_000_000):
        """P, M : profil photo (une hypothèse de tangage, cases de BIN degrés, case 0 = bord gauche) ;
        k_center : case de l'axe optique. Renvoie les k meilleurs (point, cap en degrés) et leurs votes."""
        n_head = int(round(360 / HEADING_BIN))
        votes = torch.zeros(self.n_points * n_head, dtype=torch.float32, device=device)
        queries = query_words(P, M, soft)
        # listes de lieux par paquets d'au plus `chunk` entrées : un transfert et un index_add par paquet
        spans = [(self.offsets[w], self.offsets[w + 1], w, dq) for w, dq in queries]
        spans = [(a, b, w, dq) for a, b, w, dq in spans if 0 < b - a <= max_postings]   # absent, ou trop banal
        batch, size = [], 0
        for i, sp in enumerate(spans):
            batch.append(sp)
            size += sp[1] - sp[0]
            if size >= chunk or i == len(spans) - 1:
                self._vote(votes, batch, k_center, n_head, device)
                batch, size = [], 0
        # tolérance d'un cran sur le cap : somme glissante sur 3 cases
        v = votes.view(self.n_points, n_head)
        v = v + v.roll(1, 1) + v.roll(-1, 1)
        top = torch.topk(v.view(-1), k)
        idx = top.indices.cpu().numpy()
        return idx // n_head, (idx % n_head) * HEADING_BIN, top.values.cpu().numpy()


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _vote_kernel(packed_ptr, ends_ptr, start_ptr, dq_ptr, w_ptr, votes_ptr, n_slots, total, offset,
                     N_HEAD: tl.constexpr, DIR_BITS: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
        """Un vote par entrée de l'index : chaque entrée retrouve son mot (dichotomie sur les fins cumulées),
        lit lieu et direction, calcule la case de cap et l'ajoute aux votes (même calcul que _vote)."""
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = i < total
        lo = tl.zeros([BLOCK], tl.int32)
        hi = tl.zeros([BLOCK], tl.int32) + n_slots
        for _ in range(STEPS):                                  # premier mot dont la fin cumulée dépasse i
            mid = (lo + hi) // 2
            go = tl.load(ends_ptr + mid, mask=live & (lo < hi), other=0) <= i
            lo = tl.where((lo < hi) & go, mid + 1, lo)
            hi = tl.where((lo < hi) & ~go, mid, hi)
        k = tl.minimum(lo, n_slots - 1)
        e = tl.load(packed_ptr + tl.load(start_ptr + k, mask=live, other=0) + i, mask=live, other=0)
        pt = (e >> DIR_BITS).to(tl.int64)
        d = (e & ((1 << DIR_BITS) - 1)).to(tl.float32)
        a = d - (tl.load(dq_ptr + k, mask=live, other=0.0) - offset)
        m = a % 360.0                                           # comme torch.remainder : signe du diviseur
        m = tl.where((m != 0.0) & (m < 0.0), m + 360.0, m)
        hb = m.to(tl.int64) % N_HEAD
        tl.atomic_add(votes_ptr + pt * N_HEAD + hb, tl.load(w_ptr + k, mask=live, other=0.0), mask=live, sem="relaxed")
except ImportError:                                             # pas de Triton : chemin PyTorch
    _vote_kernel = None


class GpuIndex:
    """Index entièrement sur GPU, sans perte par rapport à RidgeIndex.query :
    - seuls les mots utilisables sont gardés (les mots à plus de max_postings lieux ne sont jamais lus :
      ~64 % des entrées sur la base suisse) ;
    - lieu et direction dans un seul int32 (lieu << 9 | direction en degrés, < 512).
    Base suisse : 3,16 G entrées sur disque (18 Go) -> 1,15 G sur GPU (4,6 Go)."""

    DIR_BITS = 9

    def __init__(self, index, max_postings=2_000_000, device="cuda", chunk=50_000_000):
        assert index.n_points < 2 ** (31 - self.DIR_BITS)
        counts = np.diff(index.offsets)
        keep = counts <= max_postings
        self.offsets = np.concatenate([[0], np.cumsum(np.where(keep, counts, 0))]).astype(np.int64)
        self.offsets_t = torch.as_tensor(self.offsets, device=device)
        self.idf, self.n_points, self.device = index.idf, index.n_points, device
        self.packed = torch.empty(int(self.offsets[-1]), dtype=torch.int32, device=device)
        # segments contigus d'entrées gardées : entre deux mots écartés
        dropped = np.nonzero(~keep & (counts > 0))[0]
        bounds = [0] + [x for w in dropped for x in (index.offsets[w], index.offsets[w + 1])] + [index.offsets[-1]]
        out = 0
        for a, b in zip(bounds[0::2], bounds[1::2]):
            for i in range(a, b, chunk):
                j = min(i + chunk, b)
                pt = torch.as_tensor(np.asarray(index.point[i:j]), device=device)
                dr = torch.as_tensor(np.asarray(index.direction[i:j]).astype(np.int32), device=device)
                self.packed[out:out + j - i] = (pt << self.DIR_BITS) | dr
                out += j - i
        assert out == len(self.packed)

    def default_tile(self):
        """Tuile de votes ~ 1/4 du cache L2 (RTX 5090 : 96 Mo -> ~16 000 lieux) ; au plus 256 tuiles."""
        l2 = torch.cuda.get_device_properties(torch.device(self.device)).L2_cache_size
        n_head = int(round(360 / HEADING_BIN))
        return max(l2 // 4 // (n_head * 4), (self.n_points + 255) // 256)

    def query(self, queries, k_center, k=1000, chunk=100_000_000, tile=None):
        """queries : query_words(...). Mêmes votes que RidgeIndex.query.
        tile : lieux par tuile de votes (0 : sans tuiles ; défaut : ~1/4 du cache L2, au plus 256 tuiles)."""
        if tile is None:
            tile = self.default_tile()
        dev = self.device
        n_head = int(round(360 / HEADING_BIN))
        size = self.n_points * n_head
        shared = getattr(self, "votes", None)                # tableau partagé entre morceaux (ShardedIndex)
        votes = shared[:size].zero_() if shared is not None and len(shared) >= size else \
            torch.zeros(size, dtype=torch.float32, device=dev)
        words = torch.tensor([w for w, _ in queries], dtype=torch.long, device=dev)
        dq = torch.tensor([d for _, d in queries], dtype=torch.float32, device=dev)
        a, b = self.offsets_t[words], self.offsets_t[words + 1]
        n = b - a
        live = n > 0
        a, n, dq = a[live], n[live], dq[live]
        idf = torch.as_tensor(self.idf, device=dev)[words[live]]
        if len(n) == 0:                                      # aucun mot présent dans ce morceau
            return top_votes(votes, self.n_points, k, per_place=True)
        ends = torch.cumsum(n, 0)
        if _vote_kernel is not None and STRIDE == 1.0 and HEADING_BIN == 1.0:
            if tile:
                # Votes traités par tuiles de `tile` lieux : les écritures d'une tuile (tile x 360 x 4 octets)
                # restent dans le cache L2 au lieu de tomber au hasard dans les 2,4 Go du tableau. Dans un mot,
                # les entrées sont rangées par lieu : les limites de tuile se trouvent par dichotomie.
                n_tiles = (self.n_points + tile - 1) // tile
                bounds = torch.arange(n_tiles + 1, device=dev, dtype=torch.int64) * tile << self.DIR_BITS
                lo_i = a[:, None].expand(-1, n_tiles + 1).clone()            # (mots, tuiles + 1)
                hi_i = (a + n)[:, None].expand(-1, n_tiles + 1).clone()
                for _ in range(int(n.max()).bit_length() + 1):              # premier rang >= limite
                    mid = (lo_i + hi_i) // 2
                    below = (self.packed[mid.clamp(max=len(self.packed) - 1)].long() < bounds[None]) & (lo_i < hi_i)
                    lo_i = torch.where(below, mid + 1, lo_i)
                    hi_i = torch.where(below | (lo_i >= hi_i), hi_i, mid)
                seg_a = lo_i[:, :-1].T.reshape(-1)                         # ordre tuile par tuile
                seg_n = (lo_i[:, 1:] - lo_i[:, :-1]).T.reshape(-1)
                slot = torch.arange(len(n), device=dev).repeat(n_tiles)
                keep = seg_n > 0
                a, n, dq, idf = seg_a[keep], seg_n[keep], dq[slot[keep]], idf[slot[keep]]
                ends = torch.cumsum(n, 0)
            # un seul noyau : start[k] = début du mot k dans packed moins son rang de départ dans le paquet
            start = a - (ends - n)
            total = int(ends[-1])
            block = 1024
            _vote_kernel[(triton.cdiv(total, block),)](
                self.packed, ends, start, dq.contiguous(), idf.contiguous().float(), votes, len(n), total,
                float(k_center * BIN), N_HEAD=n_head, DIR_BITS=self.DIR_BITS,
                STEPS=24, BLOCK=block)                   # 24 pas : jusqu'à 16 M de mots, sans recompiler
            return top_votes(votes, self.n_points, k, per_place=True)
        lo = 0
        while lo < len(n):                                   # paquets d'au plus `chunk` entrées
            hi = int(torch.searchsorted(ends, ends[lo] - n[lo] + chunk, right=True))
            hi = max(hi, lo + 1)
            nn = n[lo:hi]
            rep = torch.repeat_interleave(torch.arange(hi - lo, device=dev), nn)
            start = torch.cumsum(nn, 0) - nn
            pos = a[lo:hi][rep] + (torch.arange(len(rep), device=dev) - start[rep])
            e = self.packed[pos]
            pts = (e >> self.DIR_BITS).long()
            dirs = (e & ((1 << self.DIR_BITS) - 1)).float() * STRIDE
            head = (dirs - (dq[lo:hi][rep] - k_center * BIN)) % 360
            hb = (head / HEADING_BIN).long() % n_head
            votes.index_add_(0, pts * n_head + hb, idf[lo:hi][rep])
            lo = hi
        return top_votes(votes, self.n_points, k, per_place=True)


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

    def to_gpu(self, device="cuda"):
        """Charge les index sur GPU (GpuIndex) ; les requêtes suivantes ne lisent plus la mémoire centrale."""
        self.gpu = [GpuIndex(s, device=device) for s in self.shards]
        # un seul tableau de votes pour tous les morceaux (2,4 Go chacun sinon, réalloués à chaque requête)
        votes = torch.empty(max(g.n_points for g in self.gpu) * int(round(360 / HEADING_BIN)), device=device)
        for g in self.gpu:
            g.votes = votes
        return self

    def query(self, P, M, k_center, n_places=1000, device="cuda"):
        """Renvoie jusqu'à n_places lieux distincts (indices globaux), du plus voté au moins voté."""
        pts, votes = [], []
        gpu = getattr(self, "gpu", None)
        queries = query_words(P, M) if gpu else None
        for i, (start, s) in enumerate(zip(self.starts, self.shards)):
            if gpu:
                p, _, v = gpu[i].query(queries, k_center, k=n_places)        # déjà des lieux distincts
            else:
                p, _, v = s.query(P, M, k_center, k=20 * n_places, device=device)
            pts.append(p + start)
            votes.append(v)
        pts, votes = np.concatenate(pts), np.concatenate(votes)
        order = np.argsort(-votes, kind="stable")
        _, first = np.unique(pts[order], return_index=True)
        return pts[order][np.sort(first)][:n_places]
