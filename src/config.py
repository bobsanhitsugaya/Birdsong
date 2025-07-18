import os
import yaml

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.environ.get('BIRDSONG_CONFIG', 'config.yaml')
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.cfg[key]

    def get(self, key, default=None):
        return self.cfg.get(key, default)
