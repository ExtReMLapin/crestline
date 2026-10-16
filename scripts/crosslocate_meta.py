"""Métadonnées du test CrossLocate (Tomešek et al., WACV 2022) : photos, position WGS84, champ de vision.

Le fichier .npy de CrossLocate est un pickle (dict d'objets numpy). On le lit avec un dépickleur restreint
aux seules classes numpy nécessaires : aucun code du fichier n'est exécuté.

Usage : python -I scripts/crosslocate_meta.py <Alps_photos_to_depth_compact_test.npy> <sortie.json>
"""
import io
import json
import pickle
import sys

import numpy as np

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core._multiarray_umath", "_reconstruct"), ("numpy._core._multiarray_umath", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("builtins", "dict"), ("builtins", "list"), ("collections", "OrderedDict"),
    ("_codecs", "encode"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        numpy_core = module.split(".")[:2] in (["numpy", "core"], ["numpy", "_core"])
        if (module, name) not in ALLOWED and not (numpy_core and name in ("_reconstruct", "scalar")):
            raise pickle.UnpicklingError(f"classe refusée : {module}.{name}")
        return super().find_class(module, name)


def load(path):
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read(f)
        if dtype != object:
            raise ValueError(f"tableau {dtype} inattendu")
        obj = SafeUnpickler(io.BufferedReader(f)).load()
    return obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj


def main(path, out):
    meta = load(path)
    print({k: (getattr(v, "shape", None) or len(v)) for k, v in meta.items()})
    q = [{"fn": str(fn), "lon": float(wgs[0]), "lat": float(wgs[1]), "elev": float(el),
          "yaw": float(c[0]), "pitch": float(c[1]), "roll": float(c[2]), "hfov": float(c[3])}
         for fn, wgs, el, c in zip(meta["qImageFns"], meta["qWGS"], meta["qElevations"], meta["qCameraParams"])]
    json.dump(q, open(out, "w"), indent=1)
    print(len(q), "photos ->", out)
    print(q[:3])


if __name__ == "__main__":
    main(*sys.argv[1:])
