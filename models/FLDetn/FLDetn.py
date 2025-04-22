import os
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from models.FLDetn.pkgs.ultralytics import settings, YOLO

from .model_config import ModelConfig

settings.update({'datasets_dir': './'})

def train_model(ex_dict, config: ModelConfig):
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    task = "detect"

    ex_dict['Train Results'] = ex_dict['Model'].train(
        model = f"{ex_dict['Model Name']}.yaml",
        name=name,
        data=ex_dict['Data Config'],
        project =f"{ex_dict['Output Dir']}/train",
        **config.hyperparams(allowed_keys={
                'epochs', 'time', 'patience', 'batch', 'imgsz', 'save',
                'save_period', 'cache', 'device', 'workers', 'exist_ok',
                'pretrained', 'optimizer', 'deterministic', 'single_cls', 'classes',
                'rect', 'cos_lr', 'close_mosaic', 'resume', 'amp', 'fraction',
                'profile', 'freeze', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
                'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl',
                'overlap_mask', 'mask_ratio', 'dropout', 'val', 'plots'
            })
    )
    pt_path = f"{ex_dict['Output Dir']}/train/{name}/weights/best.pt"
    ex_dict['PT path'] = pt_path
    ex_dict['Model'].load(pt_path)
    return ex_dict