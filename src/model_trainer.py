"""
Model training and cross-validation for CTR prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, f1_score
import yaml
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import gc

class CTRModelTrainer:
    """
    Handles model training, cross-validation, and hyperparameter tuning.

    Features:
    - Multiple model support
    - Automated cross-validation
    - Hyperparameter optimization
    - Model persistence
    - Performance visualization
    """

    def __init__(self, config_path: str = 'config/model_config.yaml'):
        """Initialize trainer with configuration."""
        self._setup_logging()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self._initialize_models()

    def _setup_logging(self):
        """Configure logging."""
        self.logger = logging.getLogger('CTRModelTrainer')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _initialize_models(self):
        """Initialize model instances from configuration."""
        for name, config in self.config['models'].items():
            module_path = config['class'].split('.')
            class_name = module_path[-1]
            module = __import__('.'.join(module_path[:-1]), fromlist=[class_name])
            model_class = getattr(module, class_name)

            self.models[name] = {
                'model': model_class(),
                'params': config['params'],
                'metrics': config['metrics']
            }

    def _prepare_catboost_data(self,
                              X: pd.DataFrame,
                              model_config: Dict) -> Tuple[pd.DataFrame, List[int]]:
        """
        Prepare data specifically for CatBoost, keeping categorical columns as is.

        Args:
            X: Feature DataFrame
            model_config: Model configuration dictionary

        Returns:
            Tuple of (processed DataFrame, list of categorical column indices)
        """
        X = X.copy()
        cat_features = []

        if 'categorical_features' in model_config:
            # Get indices of categorical features
            cat_features = [i for i, col in enumerate(X.columns)
                          if col in model_config['categorical_features']]

            # Ensure categorical features are of type 'str' or 'category'
            for col in model_config['categorical_features']:
                if col in X.columns:
                    X[col] = X[col].astype(str)

        return X, cat_features

    def train_evaluate_model(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_valid: pd.DataFrame,
            y_valid: pd.Series,
            model_name: str,
            custom_params: Optional[Dict] = None
    ) -> Dict:
        """
        Train and evaluate a single model using training and validation data.

        Args:
            X_train: Training features
            y_train: Training targets
            X_valid: Validation features
            y_valid: Validation targets
            model_name: Name of the model to train
            custom_params: Optional custom parameters to override defaults

        Returns:
            Dictionary containing evaluation results
        """
        gc.collect()

        self.logger.info(f"Training model: {model_name}")
        # Get model and parameters
        model_info = self.models[model_name]
        params = custom_params or model_info['params']

        # Create cross-validation splitter
        cv = StratifiedKFold(
            n_splits=self.config['cross_validation']['n_splits'],
            shuffle=True,
            random_state=42
        )

        # Combine train and validation data for cross-validation
        X_cv = pd.concat([X_train, X_valid])
        y_cv = pd.concat([y_train, y_valid])

        # Create train/validation split indices
        train_idx = range(len(X_train))
        valid_idx = range(len(X_train), len(X_cv))
        cv_splits = [(train_idx, valid_idx)]

        # Define scoring metrics
        scoring = {
            'roc_auc': make_scorer(roc_auc_score),
            'average_precision': make_scorer(average_precision_score),
            'f1': make_scorer(f1_score)
        }

        # Perform grid search
        grid_search = GridSearchCV(
            model_info['model'],
            params,
            cv=cv_splits,  # Use predefined train/validation split
            scoring=scoring,
            refit='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Fit and evaluate
        try:
            grid_search.fit(X_cv, y_cv)

            # Store best model
            self.models[model_name]['best_model'] = grid_search.best_estimator_

            # Calculate feature importance if available
            if hasattr(grid_search.best_estimator_, 'feature_importances_'):
                self.feature_importance[model_name] = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': grid_search.best_estimator_.feature_importances_
                }).sort_values('importance', ascending=False)

            # Compile results
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_,
                'feature_importance': self.feature_importance.get(model_name, None)
            }

            # Calculate additional metrics on validation set
            y_pred = grid_search.predict(X_valid)
            y_prob = grid_search.predict_proba(X_valid)[:, 1]

            results.update({
                'validation_roc_auc': roc_auc_score(y_valid, y_prob),
                'validation_average_precision': average_precision_score(y_valid, y_prob),
                'validation_f1': f1_score(y_valid, y_pred)
            })

            self.logger.info(f"Completed training for {model_name}")
            return results

        except Exception as e:
            self.logger.error(f"Error training {model_name}: {str(e)}")
            raise