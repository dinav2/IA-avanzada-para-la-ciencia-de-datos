"""
CatBoost Model Implementation
Inherits from ModelBase for consistent interface
"""

from catboost import CatBoostClassifier
from model_base import ModelBase


class CatBoostModel(ModelBase):
    def __init__(self):
        super().__init__(
            model_class=CatBoostClassifier,
            model_name="CatBoost",
            model_kwargs={
                'objective': 'Logloss',
                'eval_metric': 'Logloss',
                'bootstrap_type': 'Bernoulli',
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 50,
                'train_dir': None,
                'logging_level': 'Silent'
            }
        )
    
    def get_param_grid(self):
        """Parameter grid for hyperparameter tuning"""
        return {
            'depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'iterations': [500, 1000],
            'l2_leaf_reg': [1, 3, 5],
            'bootstrap_type': ['Bernoulli'],
            'subsample': [0.8, 0.9]
        }
    
    
    def _save_model(self, path):
        """Save CatBoost model"""
        if self.best_model is None:
            raise RuntimeError("No trained model found")
        
        self.best_model.save_model(path)
    
    def get_model_extension(self):
        """Get model file extension"""
        return "cbm"


if __name__ == "__main__":
    # Configuration
    config = {
        "TRAIN_FILE": "train_temporal.csv",
        "VALIDATION_FILE": "validation_temporal.csv", 
        "TEST_FILE": "test_temporal.csv",
        "TARGET_COL": "fraud_bool",
        "PROJECT": "fraud-detection",
        "RUN_NAME": "catboost_model_base_refactored",
        "TAGS": ["catboost", "temporal-split", "validation-tuning", "fraud-optimization"],
        "MAX_FPR": 0.1,
        "RSEED": 42
    }
    
    # Create and run model
    model = CatBoostModel()
    model.run_full_pipeline(config)
