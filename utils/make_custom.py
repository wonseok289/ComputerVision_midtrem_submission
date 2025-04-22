import argparse
import os
import yaml
import timeit
import numpy as np
import pandas as pd
import gc
import torch
from datetime import datetime
from ultralytics import YOLO
from models.FLDetn.pkgs.ultralytics import YOLO as FLDet
from models.HyperYOLOt.pkgs.hyper_ultralytics import YOLO as HyperYOLO


def parse():
    parser = argparse.ArgumentParser(
        description="Load original model's yaml and custom (depth, width)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Model Directory Path. ex)'models/YOLOv8n'"
    )
    parser.add_argument(
        "--depth", type=float, required=True,
        help="Depth"
    )
    parser.add_argument(
        "--width", type=float, required=True,
        help="Width"
    )
    return parser.parse_args()


def load_original_model_yaml(model_path, model_name):
    '''
    '''
    if model_name.lower() == "fldetn":
        temp_model = FLDet(f'models/FLDetn/pkgs/ultralytics/cfg/models/FLDet/FLDet-N.yaml')
    elif model_name.lower() == "hyperyolot":
        temp_model = HyperYOLO(f'models/HyperYOLOt/pkgs/hyper_ultralytics/cfg/models/hyper-yolo/hyper-yolot.yaml')
    else:
        # Load base model config
        temp_model = YOLO(f'{model_name}.yaml', verbose=False)
    original_model_dict = temp_model.model.yaml

    # Save original yaml
    os.makedirs(model_path, exist_ok=True)
    original_yaml_path = os.path.join(model_path, f"{model_name}_original.yaml")
    with open(original_yaml_path, 'w') as f:
        yaml.dump(original_model_dict, f, sort_keys=False)
        
    del temp_model
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return original_model_dict
    

def make_custom_yaml(model_path, model_name, depth = 0.2, width = 0.25):
    '''
    '''
    custom_depth = depth
    custom_width = width
    
    original_model_dict = load_original_model_yaml(model_path, model_name)

    # Customize depth/width and modify corresponding scale value
    custom_model_dict = original_model_dict.copy()
    scale_key = model_name.strip()[-1]
    
    # Update scale-specific values
    if 'scales' in custom_model_dict and scale_key in custom_model_dict['scales']:
        custom_model_dict['scales'][scale_key][0] = custom_depth
        custom_model_dict['scales'][scale_key][1] = custom_width

    # Also explicitly add depth_multiple and width_multiple
    custom_model_dict['depth_multiple'] = custom_depth
    custom_model_dict['width_multiple'] = custom_width

    # Save customized yaml
    custom_yaml_path = os.path.join(model_path, f"{model_name}_custom.yaml")
    with open(custom_yaml_path, 'w') as f:
        yaml.dump(custom_model_dict, f, sort_keys=False)
    
    return custom_yaml_path


def main(args):
    # data_dirs는 쉼표로 구분된 문자열을 리스트로 변환합니다.
    model_path = args.model_path
    depth = args.depth
    width = args.width

    print("입력된 모델 디렉터리:", model_path)
    print("depth:", depth)
    print("width:", width)
    
    model_name = model_path.split('/')[1].lower()
    
    # 데이터 전처리 실행
    make_custom_yaml(model_path, model_name, depth, width)

    # 추가적으로 생성된 데이터셋을 파일로 저장하거나, 후속 처리를 진행할 수 있습니다.
    print("전처리 완료.")


if __name__ == '__main__':
    args = parse()
    main(args)