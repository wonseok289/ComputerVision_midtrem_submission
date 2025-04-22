from ..base_config import BaseConfig

class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolo11n'
        self.lr0 = 0.0015
        self.dfl = 1.2
        self.cls = 0.2
        self.box = 4.
        
        self.custom_yaml_path = None # "models/YOLO11n/yolo11n_custom.yaml"
        
    def update_from_dict(self, params: dict):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(f"ModelConfig has no attribute '{k}'")
            setattr(self, k, v)