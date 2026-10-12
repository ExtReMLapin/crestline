"""Compare des segmenteurs publics (classe « ciel » ADE20K) sur CH1 et sur le test smapshot.

Usage : python scripts/eval_public_sky.py <dossier_CH1> <liste_test_smapshot> <dossier_labels> <sortie>
Mêmes mesures que eval_ch1.py / eval_smapshot.py (ligne de ciel par colonne).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).parent))
from eval_ch1 import column_errors, load_ch1  # noqa: E402
from eval_smapshot import skyline_cols  # noqa: E402

MODELS = {"segformer_b5_ade": "nvidia/segformer-b5-finetuned-ade-640-640"}


@torch.no_grad()
def sky_mask(processor, model, image):
    inputs = processor(images=Image.fromarray(image), return_tensors="pt").to("cuda")
    logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
    sky_id = {v: k for k, v in model.config.id2label.items()}["sky"]
    return (logits.argmax(1)[0] == sky_id).cpu().numpy()


def stats(e):
    e = np.concatenate(e)
    return {"median_px": float(np.median(e)), "mean_px": float(e.mean()), "p90_px": float(np.percentile(e, 90)),
            "le2px": float((e <= 2).mean()), "le5px": float((e <= 5).mean())}


def main(ch1_root, test_list, labels_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, repo in MODELS.items():
        processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForSemanticSegmentation.from_pretrained(repo).to("cuda").eval()
        ch1 = [column_errors(sky_mask(processor, model, img), gt) for _, img, gt, _ in load_ch1(ch1_root)]
        smap = []
        for sid in [l.strip() for l in open(test_list) if l.strip()]:
            z = np.load(Path(labels_dir) / f"{sid}.npz")
            smap.append(skyline_cols(sky_mask(processor, model, z["image"]), z["sky"], z["ignore"]))
        report[name] = {"ch1": stats(ch1), "smapshot_test": stats(smap)}
        for k, r in report[name].items():
            print(f"{name:18s} {k:14s} med {r['median_px']:.1f} mean {r['mean_px']:.1f} p90 {r['p90_px']:.0f} "
                  f"<=2px {r['le2px']:.3f} <=5px {r['le5px']:.3f}")
    json.dump(report, open(out_dir / "report_public.json", "w"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:5])
