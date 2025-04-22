from ..base_config import BaseConfig

class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolo12n'
        self.lr0 = 0.001
        self.dfl = 1.0
        self.cls = 0.2
        self.box = 4.5
        
        self.custom_yaml_path = None # "models/YOLO12n/yolo12n_custom.yaml"
        
    def update_from_dict(self, params: dict):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(f"ModelConfig has no attribute '{k}'")
            setattr(self, k, v)