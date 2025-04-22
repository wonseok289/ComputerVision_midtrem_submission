# tune_with_submission.py

import os
import optuna
import numpy as np
from datetime import datetime

from models.base_config import BaseConfig
from competition_utils import eval_and_vis
# 모델별 import
from models.YOLOv8n.model_config import ModelConfig
from submissions.submission_YOLOv8n import submission_YOLOv8n

# ─────────────────────────────────────────────────────
# 설정: 바꿔 가며 다른 모델도 동일 패턴으로 적용
# ─────────────────────────────────────────────────────
MODEL_NAME         = "YOLOv8n"
SubmissionFunction = submission_YOLOv8n

DATASET_NAME       = "airplane"
JSON_DIR           = os.path.join(ModelConfig.get_output_dir().replace('output', 'optuna_runs'), "json")
LABELS_DIR         = f"Datasets/{DATASET_NAME}/labels"
TOTAL_TRIALS       = 10
os.makedirs(JSON_DIR, exist_ok=True)


def sample_params(trial: optuna.Trial):
    return {
        "lr":           trial.suggest_float("lr", 1e-5, 1e-2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3),
        "momentum":     trial.suggest_float("momentum", 0.5, 0.99),
        # depth_mult, width_mult 등을 추가 가능
    }

def objective(trial: optuna.Trial) -> float:
    # 1) 하이퍼파라미터 샘플링 & Config 덮어쓰기
    params = sample_params(trial)
    cfg = ModelConfig()
    cfg.lr0           = params["lr"]
    cfg.weight_decay  = params["weight_decay"]
    cfg.momentum      = params["momentum"]
    cfg.output_dir = cfg.output_dir.replace('output', 'optuna_runs')
    
    overwrite_dict = {
        "lr0": cfg.lr0,
        "weight_decay": cfg.weight_decay,
        "momentum": cfg.momentum
    }

    # 2) 모든 split에 대해 submission → eval_and_vis
    iou_list = []
    
    for split in range(1, 11):
        yaml_path   = f"Datasets/{DATASET_NAME}/data_iter_{split:02d}.yaml"
        ex_time   = datetime.now().strftime("%y%m%d_%H%M%S")
        json_path   = os.path.join(JSON_DIR,
                          f"{ex_time}_{MODEL_NAME}_Iter_{split}_detection_results.json")
        image_level_result_path = os.path.join(JSON_DIR,
                          f"{ex_time}_{MODEL_NAME}_Iter_{split}_image_level_results.json")
        
        SubmissionFunction(yaml_path, json_path, cfg, overwrite_dict)

        # ── eval_and_vis 로 mask‑IoU 계산 ──
        stats = eval_and_vis(
            yaml_path=yaml_path,
            det_result_path=json_path,
            labels_dir=LABELS_DIR,
            image_level_result_path=image_level_result_path,  # 파일 저장 스킵
            vis_output_dir=None,
            vis=False
        )
        iou_list.append(stats["IoU"]["avg"])

    # 3) 평균 IoU 반환
    return float(np.mean(iou_list))


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"{MODEL_NAME}_cv",
        direction="maximize",
        storage=f"sqlite:///optuna_{MODEL_NAME}.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    done = len(study.trials)                          # 이미 수행된 trial 수
    remaining = TOTAL_TRIALS - done                   

    if remaining > 0:
        print(f"▶ {done} trials done, remaining {remaining}. 실행합니다.")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"▶ 이미 {done} trials를 수행했습니다. 목표({TOTAL_TRIALS})에 도달했습니다.")

    print("▶ Best hyperparameters:", study.best_trial.params)

    # 트라이얼 기록 CSV로 저장
    df = study.trials_dataframe(attrs=("number","value","params","state"))
    df.to_csv(f"optuna_{MODEL_NAME}_trials.csv", index=False)