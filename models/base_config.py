from datetime import datetime
import os
import torch

class BaseConfig:
    _experiment_time = None  # 클래스 변수로 한 번만 생성됨
    _output_dir = None
    _instance = None         # Singleton 인스턴스

    def __new__(cls, *args, **kwargs):
        # experiment_time만 공유하고, 다른 속성은 새 인스턴스로도 허용
        instance = super().__new__(cls)
        return instance

    def __init__(self, exp_time=None):
        
        # experiment_time을 한 번만 생성
        if BaseConfig._experiment_time is None:
            BaseConfig._experiment_time = exp_time or datetime.now().strftime("%y%m%d_%H%M%S")
        self.experiment_time = BaseConfig._experiment_time
        
        # output_dir 초기화
        if BaseConfig._output_dir is None:
            BaseConfig._output_dir = os.path.join("output", self.experiment_time)
            os.makedirs(BaseConfig._output_dir, exist_ok=True)
        self.output_dir = BaseConfig._output_dir
        
        self.dataset_name = 'airplane'
        self.model_name = None  # 개별 Config에서 지정
        self.exist_ok = False
        self.custom_yaml_path = None

        # Pretrained and AMP
        self.pretrained = False
        self.amp = False

        self.epochs = 20
        self.patience = 100
        self.batch = 16
        self.imgsz = 640
        self.save = True
        self.save_period = -1
        self.cache = False

        # Optimizer settings
        self.optimizer = 'AdamW'
        self.lr0 = 0.003
        self.lrf = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.warmup_epochs = 2.0
        self.warmup_momentum = 0.85
        self.warmup_bias_lr = 0.1

        # Device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.workers = 8

        # Data splits
        self.train_split = 0.6
        self.val_split = 0.2
        self.test_split = 0.2

        # Data Augmentation
        self.hsv_h = 0.015
        self.hsv_s = 0.7
        self.hsv_v = 0.4
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = 0.5
        self.flipud = 0.0
        self.fliplr = 0.5
        self.mosaic = 1.0
        self.mixup = 0.0
        self.copy_paste = 0.0

        # Loss
        self.box = 5.0
        self.cls = 0.3
        self.dfl = 1.5

        # Training strategies
        self.single_cls = True
        self.rect = False
        self.multi_scale = False
        self.cos_lr = True
        self.close_mosaic = 2
        self.resume = False
        self.fraction = 1.0
        self.profile = False
        self.freeze = None

        # Segmentation specific
        self.overlap_mask = True
        self.mask_ratio = 4

        # Classification
        self.dropout = 0.0

        # Validation
        self.val = True
        self.plots = True

    def hyperparams(self, allowed_keys=None):
        full_dict = vars(self)
        if allowed_keys is None:
            allowed_keys = {
                'epochs', 'time', 'patience', 'batch', 'imgsz', 'save',
                'save_period', 'cache', 'device', 'workers', 'exist_ok',
                'pretrained', 'optimizer', 'deterministic', 'single_cls', 'classes',
                'rect', 'multi_scale', 'cos_lr', 'close_mosaic', 'resume', 'amp', 'fraction',
                'profile', 'freeze', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
                'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl',
                'overlap_mask', 'mask_ratio', 'dropout', 'val', 'plots'
            }
        return {k: v for k, v in full_dict.items() if k in allowed_keys}
    
    @classmethod
    def get_experiment_time(cls) -> str:
        if cls._experiment_time is None:
            cls._experiment_time = datetime.now().strftime("%y%m%d_%H%M%S")
        return cls._experiment_time

    @classmethod
    def get_output_dir(cls) -> str:
        if cls._output_dir is None:
            cls._output_dir = os.path.join("output", cls.get_experiment_time())
            os.makedirs(cls._output_dir, exist_ok=True)
        return cls._output_dir