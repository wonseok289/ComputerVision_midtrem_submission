from models.base_config import BaseConfig

def update_ex_dict(ex_dict, config: BaseConfig, initial = False):
    if initial:
        ex_dict['Experiment Time'] = config.experiment_time
    ex_dict['Epochs'] = config.epochs
    ex_dict['Batch Size'] = config.batch
    ex_dict['Device'] = config.device
    ex_dict['Optimizer'] = config.optimizer
    ex_dict['LR'] = config.lr0
    ex_dict['Weight Decay'] = config.weight_decay
    ex_dict['Momentum'] = config.momentum
    ex_dict['Image Size'] = config.imgsz
    ex_dict['Output Dir'] = config.output_dir
    ex_dict['LRF'] = config.lrf      # Fimal Cosine decay learning rate
    ex_dict['Cos LR'] = config.cos_lr    # Apply Cosine Scheduler

    # Data Augmentation
    ex_dict['hsv_h'] = config.hsv_h
    ex_dict['hsv_s'] = config.hsv_s
    ex_dict['hsv_v'] = config.hsv_v
    ex_dict['degrees'] = config.degrees

    ex_dict['translate'] = config.translate
    ex_dict['scale'] = config.scale
    ex_dict['flipud'] = config.flipud
    ex_dict['fliplr'] = config.fliplr
    ex_dict['mosaic'] = config.mosaic
    ex_dict['mixup'] = config.mixup
    ex_dict['copy_paste'] = config.copy_paste
    
    ex_dict['box'] = config.box
    ex_dict['cls'] = config.cls
    ex_dict['dfl'] = config.dfl
    
    return ex_dict