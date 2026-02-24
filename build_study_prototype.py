import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="home_data/studying")
    ap.add_argument("--out_dir", type=str, default="study_artifacts")
    ap.add_argument("--max_images", type=int, default=0, help="0 = use all")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    imgs = [p for p in data_dir.rglob("*") if p.suffix.lower() in exts]
    if not imgs:
        raise RuntimeError(f"No images found in {data_dir}")

    if args.max_images and len(imgs) > args.max_images:
        imgs = imgs[: args.max_images]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Images:", len(imgs))

    # Pretrained feature extractor
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # output 512-d embedding
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    embs = []
    with torch.no_grad():
        for p in imgs:
            im = Image.open(p).convert("RGB")
            x = tfm(im).unsqueeze(0).to(device)
            e = model(x).squeeze(0)
            e = torch.nn.functional.normalize(e, dim=0)
            embs.append(e.cpu().numpy())

    embs = np.stack(embs, axis=0)  # (N, 512)

    # Prototype = mean embedding (then normalize)
    proto = embs.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-9)

    # Similarity distribution within studying set
    sims = (embs @ proto).astype(np.float32)
    # Suggest threshold: 5th percentile of studying similarities
    thr = float(np.percentile(sims, 5))

    np.save(out_dir / "study_prototype.npy", proto)
    meta = {
        "threshold": thr,
        "n_images": int(len(imgs)),
        "sim_min": float(sims.min()),
        "sim_mean": float(sims.mean()),
        "sim_p5": float(np.percentile(sims, 5)),
        "sim_p10": float(np.percentile(sims, 10)),
        "sim_p50": float(np.percentile(sims, 50)),
    }
    (out_dir / "study_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n✅ Saved:")
    print(" -", out_dir / "study_prototype.npy")
    print(" -", out_dir / "study_meta.json")
    print("\nSuggested threshold:", thr)
    print("Tip: If it misses studying too often, LOWER threshold (e.g., thr-0.03).")
    print("Tip: If it false-positives too often, RAISE threshold (e.g., thr+0.03).")


if __name__ == "__main__":
    main()