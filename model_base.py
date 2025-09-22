import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
import wandb
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, train_test_split
import time
from datetime import datetime 

class ModelBase:
    def __init__(self, model_class, model_name, model_kwargs=None):
        self.model_class = model_class
        self.model_name = model_name
        self.model = None
        self.model_path = None
        self.model_params = None
        self.model_metrics = None
        self.model_threshold = None
        
        # Hyperparameter tuning attributes
        self.best_score = -np.inf
        self.best_params = None
        self.best_model = None
        self.best_threshold = 0.5
        
        # Model-specific kwargs
        self.model_kwargs = model_kwargs or {}

    def metrics_dict(self, y_true, y_pred, y_proba):
        """Calculate metrics"""
        try:
            out = {}
            out["f1_macro"]     = f1_score(y_true, y_pred, average="macro")
            out["f1_weighted"]  = f1_score(y_true, y_pred, average="weighted")
            out["roc_auc"]      = roc_auc_score(y_true, y_proba)
            out["pr_auc"]       = average_precision_score(y_true, y_proba)
            # class 1
            out["pos_precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            out["pos_recall"]    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            out["pos_f1"]        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            # class 0 
            out["neg_precision"] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
            out["neg_recall"]    = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            out["neg_f1"]        = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            # Confusion-derived rates
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            out["tn"] = int(tn)
            out["fp"] = int(fp)
            out["fn"] = int(fn)
            out["tp"] = int(tp)
            out["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
            out["tpr"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            out["tnr"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            out["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0
            return out, cm
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Return default metrics
            return {
                "f1_macro": 0, "f1_weighted": 0, "roc_auc": 0, "pr_auc": 0,
                "pos_precision": 0, "pos_recall": 0, "pos_f1": 0,
                "neg_precision": 0, "neg_recall": 0, "neg_f1": 0,
                "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                "fpr": 1, "tpr": 0, "tnr": 0, "fnr": 1
            }, np.array([[0, 0], [0, 0]])

    def find_optimal_threshold(self, y_true, y_proba, max_fpr=0.1):
        """Find optimal threshold without full sweep"""
        thresholds = np.linspace(0.01, 0.5, 20)
        best_threshold = 0.5
        best_score = 0
        
        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            metrics, _ = self.metrics_dict(y_true, y_pred, y_proba)
            
            if metrics['fpr'] <= max_fpr:
                # Prioriza recall para detección de fraude
                score = metrics['pos_recall'] * 0.6 + metrics['pos_f1'] * 0.4
                if score > best_score:
                    best_score = score
                    best_threshold = thr
        
        return best_threshold

    def load_and_prepare_data(self, train_file, val_file, test_file, target_col):
        """Load and prepare data"""
        try:
            # Validate file paths
            for file_path in [train_file, val_file, test_file]:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load data
            df_train = pd.read_csv(train_file)
            df_val = pd.read_csv(val_file)
            df_test = pd.read_csv(test_file)
            
            # Validate target column exists
            for df, name in [(df_train, "train"), (df_val, "validation"), (df_test, "test")]:
                if target_col not in df.columns:
                    raise ValueError(f"Target column '{target_col}' not found in {name} data")
            
            # Feature alignment
            feat_train = [c for c in df_train.columns if c != target_col]
            if len(feat_train) == 0:
                raise ValueError("No features found in training data")
            
            X_train = df_train[feat_train]
            y_train = df_train[target_col]
            
            # Align validation and test sets
            X_val = df_val.reindex(columns=feat_train, fill_value=0)
            y_val = df_val[target_col]
            X_test = df_test.reindex(columns=feat_train, fill_value=0)
            y_test = df_test[target_col]
            
            print(f"Data loaded successfully:")
            print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"  Validation: {X_val.shape[0]} samples")
            print(f"  Test: {X_test.shape[0]} samples")
            
            return X_train, y_train, X_val, y_val, X_test, y_test, feat_train
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, param_grid, max_combinations=8):
        """Hyperparameter tuning"""
        try:
            # Validate inputs
            if not isinstance(param_grid, dict):
                raise ValueError("param_grid must be a dictionary")
            
            if max_combinations <= 0:
                raise ValueError("max_combinations must be positive")
            
            # Initialize best tracking variables
            self.best_score = -np.inf
            self.best_params = None
            self.best_model = None
            self.best_threshold = 0.5
            
            # Use 20% of training data for tuning
            X_train_sub, _, y_train_sub, _ = train_test_split(
                X_train, y_train, 
                train_size=0.2,
                random_state=42, 
                stratify=y_train
            )
            
            total_combinations = min(max_combinations, len(list(ParameterGrid(param_grid))))
            print(f"Testing {total_combinations} parameter combinations...")
            
            tuning_results = []
            start_time = time.time()
            
            for i, params in enumerate(ParameterGrid(param_grid)):
                if i >= total_combinations:
                    break
                
                try:
                    # Create model - avoid duplicate parameters
                    # model_kwargs already contains model-specific defaults
                    model_params = {**self.model_kwargs, **params}
                    model = self.model_class(**model_params)
                    
                    # Train with early stopping
                    try:
                        # Try with eval_set first (for LightGBM, XGBoost, CatBoost)
                        model.fit(
                            X_train_sub, y_train_sub,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    except TypeError:
                        # Fallback for models that don't support eval_set
                        model.fit(X_train_sub, y_train_sub)
                    
                    # Get validation predictions
                    val_proba = model.predict_proba(X_val)[:, 1]
                    threshold = self.find_optimal_threshold(y_val, val_proba)
                    
                    # Evaluate with optimal threshold
                    val_pred = (val_proba >= threshold).astype(int)
                    metrics, _ = self.metrics_dict(y_val, val_pred, val_proba)
                    
                    # Score: prioritize recall for fraud detection
                    score = metrics['pos_recall'] * 0.6 + metrics['pos_f1'] * 0.4
                    
                    tuning_results.append({
                        'params': params.copy(),
                        'val_score': score,
                        'val_threshold': threshold,
                        'val_fpr': metrics['fpr'],
                        'val_tpr': metrics['tpr'],
                        'val_f1': metrics['pos_f1']
                    })
                    
                    # Update best if better
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params.copy()
                        self.best_threshold = threshold
                        self.best_model = model
                    
                    print(f"Combination {i+1}/{total_combinations}: Score={score:.4f}, Threshold={threshold:.4f}, FPR={metrics['fpr']:.4f}")
                    
                except Exception as e:
                    print(f"Warning: Failed to train model with params {params}: {e}")
                    continue
            
            tuning_time = time.time() - start_time
            print(f"Hyperparameter tuning completed in {tuning_time/60:.1f} minutes")
            print(f"Best score: {self.best_score:.4f}")
            print(f"Best threshold: {self.best_threshold:.4f}")
            
            if self.best_params is None:
                raise RuntimeError("No successful model training during hyperparameter tuning")
            
            return tuning_results
            
        except Exception as e:
            print(f"Error in hyperparameter tuning: {e}")
            raise

    def train_final_model(self, X_train, y_train, X_val, y_val):
        """Train the final model with all data"""
        try:
            if self.best_params is None:
                raise RuntimeError("No best parameters found. Run hyperparameter_tuning first.")
            
            # Create final model - avoid duplicate parameters
            # model_kwargs already contains model-specific defaults
            final_params = {**self.model_kwargs, **self.best_params}
            final_model = self.model_class(**final_params)
            
            start_time = time.time()
            try:
                # Try with eval_set first (for LightGBM, XGBoost, CatBoost)
                final_model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
            except TypeError:
                # Fallback for models that don't support eval_set
                final_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            print(f"Final model training completed in {training_time/60:.1f} minutes")
            
            # Update best model
            self.best_model = final_model
            return training_time
            
        except Exception as e:
            print(f"Error training final model: {e}")
            raise

    def evaluate_model(self, X_test, y_test, max_fpr=0.1):
        """Evaluate the model on the test set"""
        try:
            if self.best_model is None:
                raise RuntimeError("No trained model found. Run train_final_model first.")
            
            if max_fpr <= 0 or max_fpr > 1:
                raise ValueError("max_fpr must be between 0 and 1")
            
            # Generate predictions
            test_proba = self.best_model.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold on test set
            test_threshold = self.find_optimal_threshold(y_test, test_proba, max_fpr)
            
            # Evaluate with optimal threshold
            test_pred = (test_proba >= test_threshold).astype(int)
            metrics, cm = self.metrics_dict(y_test, test_pred, test_proba)
            
            print(f"Test evaluation completed")
            print(f"   Threshold: {test_threshold:.4f}")
            print(f"   F1: {metrics['pos_f1']:.4f}, Precision: {metrics['pos_precision']:.4f}")
            print(f"   Recall: {metrics['pos_recall']:.4f}, FPR: {metrics['fpr']:.4f}")
            
            return {
                'test_proba': test_proba,
                'test_pred': test_pred,
                'test_threshold': test_threshold,
                'metrics': metrics,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            raise

    def run_full_pipeline(self, config):
        """Run the full pipeline"""
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test, feat_train = self.load_and_prepare_data(
            config["TRAIN_FILE"], config["VALIDATION_FILE"], config["TEST_FILE"], config["TARGET_COL"]
        )
        
        # Hyperparameter tuning
        param_grid = self.get_param_grid()
        tuning_results = self.hyperparameter_tuning(X_train, y_train, X_val, y_val, param_grid)
        
        # Train final model
        training_time = self.train_final_model(X_train, y_train, X_val, y_val)
        
        # Initialize WandB
        wandb.init(
            project=config["PROJECT"],
            name=config["RUN_NAME"],
            tags=config["TAGS"],
            config={
                "model": self.model_name,
                "seed": config["RSEED"],
                "optimal_threshold_validation": self.best_threshold,
                "max_fpr_constraint": config["MAX_FPR"],
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "n_features": len(feat_train),
                "best_validation_score": self.best_score,
                "final_model_training_time_minutes": training_time/60,
                **self.best_params
            },
            save_code=True,
            reinit=True,
        )
        
        # Log tuning results
        wandb.log({"tables/hyperparameter_tuning": wandb.Table(dataframe=pd.DataFrame(tuning_results))})
        
        # Evaluate model
        results = self.evaluate_model(X_test, y_test, config["MAX_FPR"])
        
        # Log results to WandB
        wandb.log({f"test_optimal/{k}": v for k, v in results['metrics'].items() if isinstance(v, (int, float))})
        self.log_curves_to_wandb(y_test, results['test_proba'])
        self.log_confusion_matrix(y_test, results['test_pred'])
        
        # Save artifacts
        model_path = f"{self.model_name.lower()}_model.{self.get_model_extension()}"
        self.save_artifacts(model_path, feat_train)
        
        wandb.finish()
        print(f"\n🎊 {self.model_name} EXPERIMENT COMPLETED SUCCESSFULLY! 🎊")
    
    def log_curves_to_wandb(self, y_true, y_proba):
        """Log ROC and PR curves to WandB"""
        try:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = roc_auc_score(y_true, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            wandb.log({"roc_curve": wandb.Image(plt)})
            plt.close()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            wandb.log({"pr_curve": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not log curves to WandB: {e}")
    
    def log_confusion_matrix(self, y_true, y_pred):
        """Log confusion matrix to WandB"""
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            classes = ['Non-Fraud', 'Fraud']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            wandb.log({"confusion_matrix": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            print(f"Could not log confusion matrix to WandB: {e}")
    
    def save_artifacts(self, model_path, feat_train):
        """Save model only"""
        try:
            # Save model
            self._save_model(model_path)
            print(f"Model saved to: {model_path}")
            
            # Try to log model to WandB
            try:
                wandb.save(model_path)
                print("Model logged to WandB successfully")
            except OSError as e:
                    raise
            
        except Exception as e:
            print(f"Error saving artifacts: {e}")
            raise
    
    # Abstract methods that must be implemented by subclasses
    def _save_model(self, path):
        """Abstract method to save specific model"""
        raise NotImplementedError("Subclasses must implement _save_model")
    
    
    def get_param_grid(self):
        """Abstract method to get parameter grid"""
        raise NotImplementedError("Subclasses must implement get_param_grid")
    
    def get_model_extension(self):
        """Abstract method to get model file extension"""
        raise NotImplementedError("Subclasses must implement get_model_extension")