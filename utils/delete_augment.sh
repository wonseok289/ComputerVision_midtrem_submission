#!/usr/bin/env bash
# delete_augmented.sh
# Usage: ./delete_augmented.sh <dataset_name>

if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

DATASET="$1"
BASE="Datasets/${DATASET}"
IMG_DIR="${BASE}/images"
LBL_DIR="${BASE}/labels"

echo ">> Deleting augmented image files (_fliplr, _aug)…"
find "$IMG_DIR" -type f \( -iname "*_fliplr.*" -o -iname "*_aug.*" \) -print -delete

echo ">> Deleting augmented label files (_fliplr.txt, _aug.txt)…"
find "$LBL_DIR" -type f \( -iname "*_fliplr.txt" -o -iname "*_aug.txt" \) -print -delete

echo ">> Cleaning split .txt files (removing lines with _fliplr or _aug)…"
for SPLIT in "${BASE}/train_iter_"*".txt" "${BASE}/val_iter_"*".txt" "${BASE}/test_iter_"*".txt"; do
  [ -f "$SPLIT" ] || continue
  echo "   - Processing $(basename "$SPLIT")"
  # 백업(.bak) 남기고, 해당 패턴이 있는 줄 삭제
  sed -i.bak '/_fliplr\|_aug/d' "$SPLIT"
  rm -f "${SPLIT}.bak"
done

echo ">> Done."
