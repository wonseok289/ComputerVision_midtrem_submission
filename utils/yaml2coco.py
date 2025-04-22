# utils/yaml2coco_txtsplit.py  (최종 안정판)

import yaml, json, cv2, argparse
from pathlib import Path
from tqdm import tqdm

CATEGORY_ID, CATEGORY_NAME = 0, "airplane"

# ---------- 기본 유틸 ----------
def yolo2coco_box(xywh_norm, w_img, h_img):
    xc, yc, w, h = map(float, xywh_norm)
    w *= w_img;  h *= h_img
    x = xc * w_img - w / 2
    y = yc * h_img - h / 2
    return [x, y, w, h]

def base_coco():
    return {
        "info": {"description": "YOLO txt → COCO json"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}]
    }

# ---------- 경로 해결 ----------
def resolve_path(root: Path, raw: str) -> Path | None:
    """
    txt 라인(raw)을 실제 파일경로로 변환
    1) 절대경로 그대로
    2) root / raw
    3) root.parent / raw          <-- 'Datasets/airplane/…' 중복 문제 해결
    4) cwd / raw
    """
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for base in (root, root.parent, Path.cwd()):
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return None

# ---------- 변환 루틴 ----------
def txt_to_coco(root, txt_file, tag):
    root = Path(root).resolve()
    coco, ann_id = base_coco(), 1

    with open(txt_file) as f:
        img_lines = [ln.strip() for ln in f if ln.strip()]

    for img_id, raw in enumerate(tqdm(img_lines, desc=tag)):
        img_path = resolve_path(root, raw)
        if img_path is None:
            print(f"⚠️  경로 해결 실패: {raw}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  이미지 로드 실패: {img_path}")
            continue
        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": str(img_path.relative_to(root)),
            "width": w,
            "height": h
        })

        lbl = root / "labels" / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        with open(lbl) as lf:
            for ln in lf:
                cls, *box = ln.split()
                coco_box = yolo2coco_box(box, w, h)
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CATEGORY_ID,
                    "bbox": coco_box,
                    "area": coco_box[2] * coco_box[3],
                    "iscrowd": 0
                })
                ann_id += 1
    return coco

def convert_yaml(yaml_path: Path, out_dir: Path):
    cfg = yaml.safe_load(yaml_path.read_text())
    root = Path(cfg["path"]).resolve()
    tag  = yaml_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        txt_file = resolve_path(root, cfg[split])
        if txt_file is None:
            print(f"⚠️  {cfg[split]} 찾을 수 없어 건너뜀")
            continue
        coco = txt_to_coco(root, txt_file, f"{split}_{tag}")
        (out_dir / f"{split}_{tag}.json").write_text(json.dumps(coco, indent=4))
        print(f"✓  {split}_{tag}.json 저장")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser("Ultralytics YAML(txt split) → COCO JSON 변환기")
    ap.add_argument("--root", required=True, help="데이터 루트 (YAML이 있는 곳)")
    ap.add_argument("--pattern", default="data_iter_*.yaml", help="YAML glob 패턴")
    ap.add_argument("--out", default=None, help="JSON 저장 폴더 (기본 root/annotations)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out  = Path(args.out) if args.out else root / "annotations"
    ymls = sorted(root.glob(args.pattern))
    if not ymls:
        print("⚠️  매칭되는 YAML이 없습니다."); return
    for y in ymls:
        convert_yaml(y, out)

if __name__ == "__main__":
    main()
