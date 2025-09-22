"""
XGBoost Model Implementation
Inherits from ModelBase for consistent interface
"""

import xgboost as xgb
from xgboost import XGBClassifier
from model_base import ModelBase


class XGBoostModel(ModelBase):
    def __init__(self):
        super().__init__(
            model_class=XGBClassifier,
            model_name="XGBoost",
            model_kwargs={
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'booster': 'gbtree',
                'random_state': 42,
                'verbosity': 0
            }
        )
    
    def get_param_grid(self):
        """Parameter grid for hyperparameter tuning"""
        return {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3]
        }
    
    
    def _save_model(self, path):
        """Save XGBoost model"""
        if self.best_model is None:
            raise RuntimeError("No trained model found")
        
        self.best_model.save_model(path)
    
    def get_model_extension(self):
        """Get model file extension"""
        return "json"


if __name__ == "__main__":
    # Configuration
    config = {
        "TRAIN_FILE": "train_temporal.csv",
        "VALIDATION_FILE": "validation_temporal.csv", 
        "TEST_FILE": "test_temporal.csv",
        "TARGET_COL": "fraud_bool",
        "PROJECT": "fraud-detection",
        "RUN_NAME": "xgb_model",
        "TAGS": ["xgboost", "temporal-split", "validation-tuning", "fraud-optimization"],
        "MAX_FPR": 0.1,
        "RSEED": 42
    }
    
    # Create and run model
    model = XGBoostModel()
    model.run_full_pipeline(config)
