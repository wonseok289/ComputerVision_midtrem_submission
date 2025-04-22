import os
import glob
import cv2
import random
import numpy as np

import random
import cv2
import numpy as np

def apply_gridmask(
    img: np.ndarray,
    prob: float = 0.5,
    num_grid: int = 4,
    ratio: float = 0.25,
    shift: int = 10,
    rotate: float = 10
) -> np.ndarray:
    """
    Apply GridMask augmentation with square cells to a single image.

    Args:
        img (np.ndarray): H×W×C uint8 image.
        prob (float): probability of applying the mask (0.0–1.0).
        num_grid (int): number of cells per axis (creates num_grid×num_grid grid).
        ratio (float): fraction of each cell to mask (hole size = ratio × cell_size).
        shift (int): maximum pixel shift for hole origin right/down (0–shift).
        rotate (float): maximum rotation angle in degrees (±rotate).

    Returns:
        np.ndarray: augmented image (same shape and dtype as input).
    """
    if random.random() >= prob:
        return img

    h, w = img.shape[:2]
    # Ensure square cells: use the larger dimension to define cell size
    cell_size = int(np.ceil(max(h, w) / num_grid))
    hole_size = int(cell_size * ratio + 0.5)
    off_h = random.randint(0, shift)
    off_w = random.randint(0, shift)

    mask = np.ones((h, w), dtype=np.uint8)
    # Carve square holes in each cell
    for i in range(num_grid):
        for j in range(num_grid):
            y = i * cell_size + off_h
            x = j * cell_size + off_w
            y1 = y + (cell_size - hole_size) // 2
            x1 = x + (cell_size - hole_size) // 2
            y2 = y1 + hole_size
            x2 = x1 + hole_size
            y1c, x1c = max(0, y1), max(0, x1)
            y2c, x2c = min(h, y2), min(w, x2)
            if y1c < y2c and x1c < x2c:
                mask[y1c:y2c, x1c:x2c] = 0

    # Rotate mask
    if rotate:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-rotate, rotate), 1.0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # Apply mask (black holes)
    if img.ndim == 3:
        mask = mask[:, :, None]
    return img * mask

def apply_color_jitter(
    img: np.ndarray,
    prob: float = 0.5, 
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1
) -> np.ndarray:
    """
    Randomly jitter brightness, contrast, saturation, and hue of an image.

    Args:
        img (np.ndarray): Input image in BGR uint8 format.
        brightness (float): max brightness change as a fraction of 255 (±).
        contrast (float): max contrast multiplier change (±).
        saturation (float): max saturation multiplier change (±).
        hue (float): max hue shift as fraction of 180° (±).

    Returns:
        np.ndarray: Color-jittered image in BGR uint8.
    """
    
    if random.random() >= prob:
        return img
    
    # Convert to float32 for more precise operations
    out = img.astype(np.float32)

    # 1) Brightness & Contrast (on BGR)
    #    contrast α ∈ [1-contrast, 1+contrast]
    α = random.uniform(1 - contrast, 1 + contrast)
    #    brightness β ∈ [-brightness*255, +brightness*255]
    β = random.uniform(-brightness * 255, brightness * 255)
    out = α * out + β
    out = np.clip(out, 0, 255)

    # 2) Saturation & Hue (in HSV)
    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    #    hue shift ∈ [−hue*180, +hue*180]
    h = (h + random.uniform(-hue * 180, hue * 180)) % 180
    #    saturation scale ∈ [1-saturation, 1+saturation]
    s *= random.uniform(1 - saturation, 1 + saturation)
    #    clip saturation
    s = np.clip(s, 0, 255)
    #    value (brightness) channel already adjusted above, but you can also jitter here:
    v *= random.uniform(1-brightness, 1+brightness)
    v = np.clip(v, 0, 255)

    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return out

def augment_dataset(dataset_name: str):
    base_dir = os.path.join("Datasets", dataset_name)
    img_dir  = os.path.join(base_dir, "images")
    lbl_dir  = os.path.join(base_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    split_files = glob.glob(os.path.join(base_dir, "train_iter_*.txt"))

    # 이미 증강된 데이터가 하나라도 있으면 스킵
    already = any(
        glob.glob(os.path.join(img_dir, f"*{suffix}{ext}"))
        for suffix in ("_aug",)
        for ext in exts
    )
    if already:
        print("✅ 데이터셋이 이미 증강되어 있습니다. 작업을 건너뜁니다.")
        return
    

    for img_path in glob.glob(os.path.join(img_dir, "*")):
        stem, ext = os.path.splitext(os.path.basename(img_path))
        if ext.lower() not in exts:
            continue

        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        if not os.path.isfile(lbl_path):
            print(f"⚠️ 레이블 없음, 스킵: {lbl_path}")
            continue

        # 원본 불러와 좌우 뒤집기
        img      = cv2.imread(img_path)
        aug_img  = cv2.flip(img, 1)

        # 확률적으로 GridMask 적용
        aug_img =apply_gridmask(aug_img, prob=0.5)
        aug_img =apply_color_jitter(aug_img, prob=0.5)

        # 레이블 읽어 x_center 뒤집기
        new_lines = []
        with open(lbl_path, "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = line.strip().split()
                xc = 1.0 - float(xc)
                new_lines.append(f"{cls} {xc:.6f} {yc} {bw} {bh}")

        # 파일명 _aug
        out_img   = os.path.join(img_dir,   f"{stem}_aug{ext}")
        out_lbl   = os.path.join(lbl_dir,   f"{stem}_aug.txt")
        orig_line = os.path.join(base_dir, "images", f"{stem}{ext}")
        aug_line  = os.path.join(base_dir, "images", f"{stem}_aug{ext}")

        # 저장
        cv2.imwrite(out_img, aug_img)
        with open(out_lbl, "w") as wf:
            wf.write("\n".join(new_lines) + "\n")
        print(f"➕ 생성됨: {stem}_aug")

        # train split 파일에만 추가
        for split_file in split_files:
            with open(split_file, "r") as rf:
                lines = {l.strip() for l in rf if l.strip()}
            if orig_line in lines:
                with open(split_file, "a") as af:
                    af.write(aug_line + "\n")

    # ── 요약 ──
    originals = len([p for p in glob.glob(os.path.join(img_dir, "*"))
                     if os.path.splitext(p)[1].lower() in exts and not p.endswith("_aug" + os.path.splitext(p)[1])])
    augs = len([p for p in glob.glob(os.path.join(img_dir, "*_aug*"))
                 if os.path.splitext(p)[1].lower() in exts])
    print(f"\n완료: 원본 {originals}장 → aug {augs}장 (총 {originals + augs}장)")
