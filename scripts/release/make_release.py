"""Prépare le dossier de publication Hugging Face de Crestline (local uniquement, rien n'est envoyé).

Usage : python scripts/release/make_release.py <dossier_run> <dossier_photos> <sortie> <rapport_ch1.json> <rapport_smapshot.json>

- poids fp16 en safetensors + config.json
- images de la fiche régénérées avec les poids publiés et crestline.py (teste le chemin de publication),
  sans métadonnées EXIF
- README.md (fiche du modèle) avec les métriques lues dans les rapports d'évaluation
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

HERE = Path(__file__).parent
DEMO = "IMG_3373"
LONG_SIDE = 1600


def metrics(ch1_report, smap_report):
    ch1_ours = json.load(open(ch1_report))
    smap_ours = json.load(open(smap_report))
    ch1_all = json.load(open("outputs/eval_ch1_all/report_ch1.json"))
    smap_all = json.load(open("outputs/eval_smapshot_clean/report.json"))
    public = json.load(open("outputs/eval_public/report_public.json"))
    ours_key = next(k for k in ch1_ours if k not in ("mask2former", "n_images"))
    assert ours_key in smap_ours, (ours_key, list(smap_ours))
    rows = [
        ("**Crestline (this model)**", ch1_ours[ours_key], smap_ours[ours_key], smap_ours[ours_key]["ridge_f1"]),
        ("SegFormer-B5 (ADE20K, sky class)", public["segformer_b5_ade"]["ch1"], public["segformer_b5_ade"]["smapshot_test"], None),
        ("Mask2Former Swin-L (ADE20K, sky class)", ch1_all["mask2former"], smap_all["mask2former"], None),
        ("Same pipeline trained on GeoPose3K (ConvNeXt-Large)", ch1_all["ridge_C_large512"], smap_all["ridge_C_large512"], smap_all["ridge_C_large512"]["ridge_f1"]),
    ]
    return rows, ch1_ours["n_images"], smap_ours["n_images"]


def table(rows):
    out = ["| Model | CH1 mean error | CH1 ≤ 2 px | smapshot-test mean error | smapshot-test ≤ 2 px | Inner-ridge F1 |",
           "|---|---|---|---|---|---|"]
    for name, ch1, smap, f1 in rows:
        f1s = f"**{f1:.2f}**" if (f1 is not None and name.startswith("**")) else (f"{f1:.2f}" if f1 is not None else "—")
        b = "**" if name.startswith("**") else ""
        out.append(f"| {name} | {b}{ch1['mean_px']:.1f} px{b} | {b}{100 * ch1['le2px']:.1f} %{b} | "
                   f"{b}{smap['mean_px']:.1f} px{b} | {b}{100 * smap['le2px']:.1f} %{b} | {f1s} |")
    return "\n".join(out)


def check_equivalence(model, run_dir, photo_dir):
    """Refuse la publication si les poids publiés ne reproduisent pas le modèle d'origine (fp32)."""
    import crestline
    import numpy as np
    import segmentation_models_pytorch as smp
    cfg = json.load(open(run_dir / "config.json"))
    ref = smp.Unet(cfg["encoder"], encoder_weights=None, classes=3)
    ref.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    ref = ref.to(next(model.parameters()).device).eval()
    for path in sorted(Path(photo_dir).glob("*.[jJ][pP][gG]"))[:3]:
        _, p_ref = crestline.predict(ref, Image.open(path))
        _, p_pub = crestline.predict(model, Image.open(path))
        sky_diff = float(np.mean((p_ref[0] > 0.5) != (p_pub[0] > 0.5)))
        assert sky_diff < 0.001, f"poids publiés non équivalents sur {path.name} : {sky_diff:.1%} de pixels de ciel différents"
        print(f"équivalence OK sur {path.name} (ciel différent : {sky_diff:.4%}, écart max crêtes "
              f"{np.abs(p_ref[2] - p_pub[2]).max():.3f})")


def main(run_dir, photo_dir, out_dir, ch1_report, smap_report):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    # poids et configuration
    state = torch.load(run_dir / "best.pt", map_location="cpu")
    # bf16 et non fp16 : ~110 000 poids ConvNeXt sont sous la précision du fp16 et le modèle s'effondre
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in state.items()}, out_dir / "model.safetensors")
    train_cfg = json.load(open(run_dir / "config.json"))
    thr = json.load(open(run_dir / "ridge_threshold.json"))["best"]
    json.dump({"architecture": "smp.Unet", "encoder": train_cfg["encoder"], "classes": ["sky", "skyline", "inner_ridges"],
               "ridge_threshold": float(thr), "input": "RGB, ImageNet normalisation, any size (padded to /32)",
               "n_train_images": train_cfg["n_train"]}, open(out_dir / "config.json", "w"), indent=2)
    shutil.copy(HERE / "crestline.py", out_dir / "crestline.py")
    # images de la fiche, avec le chemin de publication (poids fp16 + crestline.py)
    sys.path.insert(0, str(out_dir))
    sys.dont_write_bytecode = True
    import crestline  # noqa: E402
    model, config = crestline.load(out_dir)
    check_equivalence(model, run_dir, photo_dir)
    gallery = []
    for path in sorted(Path(photo_dir).glob("*.[jJ][pP][gG]")):
        img, prob = crestline.predict(model, Image.open(path), long_side=2048)
        rows, ridges = crestline.lines(prob, config["ridge_threshold"])
        vis = crestline.overlay(img, rows, ridges, width=5)
        vis.thumbnail((LONG_SIDE, LONG_SIDE), Image.LANCZOS)
        name = "demo.jpg" if path.stem == DEMO else f"{path.stem.lower()}.jpg"
        Image.frombytes(vis.mode, vis.size, vis.tobytes()).save(out_dir / "images" / name, quality=88)  # sans EXIF
        if path.stem != DEMO:
            gallery.append(name)
    rows, n_ch1, n_smap = metrics(ch1_report, smap_report)
    card = (HERE / "README_template.md").read_text()
    card = card.replace("{{TABLE}}", table(rows)).replace("{{N_CH1}}", str(n_ch1)).replace("{{N_SMAP}}", str(n_smap))
    card = card.replace("{{N_TRAIN}}", f"{train_cfg['n_train']:,}").replace("{{RIDGE_THR}}", f"{thr}")
    card = card.replace("{{GALLERY}}", "\n".join(f"![](images/{g})" for g in gallery))
    (out_dir / "README.md").write_text(card)
    print("prêt :", out_dir, sorted(p.name for p in out_dir.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main(*sys.argv[1:6])
