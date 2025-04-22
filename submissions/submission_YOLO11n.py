import gc
import os
import cv2
import yaml
import torch
import random
import numpy as np
from PIL import Image
from datetime import datetime
from models import YOLO11n
from utils.ex_dict import update_ex_dict
from utils.offline_augmentation import augment_dataset
from utils.make_custom import make_custom_yaml


def submission_YOLO11n(yaml_path, output_json_path, config = None):
    
    ###### can be modified (Only Hyperparameters, which can be modified in demo) ######
    hyperparams = {
        'model_name': 'yolo11n',
        'epochs': 20,
        'batch': 16,
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'optimizer': 'AdamW',
        'dfl': 1.5,
        'cls': 0.5,
        'box': 7.5,
        'close_mosaic': 2,
        'cos_lr': True,
        'custom_yaml_path': None,
    }
    
    depth = 0.50
    width = 0.25
    conf = 0.25
    enable_tta = True
    
    if config is None:
        config = YOLO11n.ModelConfig()
    config.update_from_dict(hyperparams)
    data_config = load_yaml_config(yaml_path)
    ex_dict = {}
    ex_dict = update_ex_dict(ex_dict, config, initial=True)
    
    ###### can be modified (Only Models, which can't be modified in demo) ######
    from ultralytics import YOLO
    ex_dict['Iteration']  = int(yaml_path.split('.yaml')[0][-2:])
    
    if ex_dict['Iteration'] == 1 and hyperparams['custom_yaml_path'] is not None:
        make_custom_yaml(model_path="models/YOLO11n", model_name='yolo11n', depth=depth, width=width)
    
    Dataset_Name = yaml_path.split('/')[1]
    
    ex_dict['Dataset Name'] = Dataset_Name
    ex_dict['Data Config'] = yaml_path
    ex_dict['Number of Classes'] = data_config['nc']
    ex_dict['Class Names'] = data_config['names']
    
    augment_dataset(Dataset_Name)
    control_random_seed(42)
    
    if config.custom_yaml_path is not None:
        model_yaml_path = config.custom_yaml_path
    else:
        model_yaml_path = f'{config.model_name}.yaml'
    
    model = YOLO(model_yaml_path, verbose=False)
    os.makedirs(config.output_dir, exist_ok=True)
    
    ex_dict['Model Name'] = config.model_name
    ex_dict['Model'] = model
    
    ex_dict = YOLO11n.train_model(ex_dict, config)
    
    test_images = get_test_images(data_config)
    results_dict = detect_and_save_bboxes(ex_dict['Model'], test_images, conf, enable_tta)
    save_results_to_file(results_dict, output_json_path)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_test_images(config):
    test_path = config['test']
    root_path = config['path']

    test_path = os.path.join(root_path, test_path)
    
    if os.path.isdir(test_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        for root, _, files in os.walk(test_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    elif test_path.endswith('.txt'):
        with open(test_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        return image_paths
    
    
def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False 


def detect_and_save_bboxes(model, image_paths, conf, enable_tta):
    results_dict = {}

    for img_path in image_paths:
        results = model(img_path, verbose=False, conf=conf, task='detect', augment=enable_tta)
        img_results = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                img_results.append({
                    'bbox': bbox,  # [x1, y1, x2, y2]
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        results_dict[img_path] = img_results
    return results_dict


def save_results_to_file(results_dict, output_path):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"결과가 {output_path}에 저장되었습니다.")