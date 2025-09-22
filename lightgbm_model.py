"""
LightGBM Model Implementation
Inherits from ModelBase for consistent interface
"""

import lightgbm as lgb
from lightgbm import LGBMClassifier
from model_base import ModelBase


class LightGBMModel(ModelBase):
    def __init__(self):
        super().__init__(
            model_class=LGBMClassifier,
            model_name="LightGBM",
            model_kwargs={
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'bagging_freq': 5,
                'verbosity': -1,
                'random_state': 42
            }
        )
    
    def get_param_grid(self):
        """Parameter grid for hyperparameter tuning"""
        return {
            'num_leaves': [31, 50],
            'learning_rate': [0.05, 0.1],
            'feature_fraction': [0.8, 0.9],
            'bagging_fraction': [0.8, 0.9],
            'min_data_in_leaf': [20, 50],
            'max_depth': [6, 8]
        }
    
    
    def _save_model(self, path):
        """Save LightGBM model"""
        if self.best_model is None:
            raise RuntimeError("No trained model found")
        
        self.best_model.booster_.save_model(path)
    
    def get_model_extension(self):
        """Get model file extension"""
        return "txt"


if __name__ == "__main__":
    # Configuration
    config = {
        "TRAIN_FILE": "train_temporal.csv",
        "VALIDATION_FILE": "validation_temporal.csv", 
        "TEST_FILE": "test_temporal.csv",
        "TARGET_COL": "fraud_bool",
        "PROJECT": "fraud-detection",
        "RUN_NAME": "lgb_model",
        "TAGS": ["lightgbm", "temporal-split", "validation-tuning", "fraud-optimization"],
        "MAX_FPR": 0.1,
        "RSEED": 42
    }
    
    # Create and run model
    model = LightGBMModel()
    model.run_full_pipeline(config)
