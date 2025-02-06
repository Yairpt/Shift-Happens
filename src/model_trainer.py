"""
Model training and cross-validation for CTR prediction.
"""
import itertools

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, f1_score
import yaml
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from xgboost import XGBClassifier, XGBRegressor


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

            if class_name == 'XGBClassifier':
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    objective='binary:logistic',
                    use_label_encoder=False,
                    enable_categorical=True
                )
            elif class_name == 'CatBoostClassifier':
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(
                    verbose=False,
                    allow_writing_files=False
                )
            elif class_name == 'LGBMClassifier':
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    objective='binary',
                    verbose=-1
                )
            else:
                module = __import__('.'.join(module_path[:-1]), fromlist=[class_name])
                model_class = getattr(module, class_name)
                model = model_class()

            self.models[name] = {
                'model': model,
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

    def train_evaluate_model(self, X_train, y_train, X_valid, y_valid, model_name, custom_params=None):
        gc.collect()
        model_info = self.models[model_name]
        params = custom_params or model_info['params']

        scoring = {
            'roc_auc': make_scorer(roc_auc_score),
            'average_precision': make_scorer(average_precision_score),
            'f1': make_scorer(f1_score)
        }

        if isinstance(model_info['model'], (XGBClassifier, LGBMClassifier, CatBoostClassifier)):
            best_score = float('-inf')
            best_model = None
            best_params = None

            param_combinations = [dict(zip(params.keys(), v))
                                  for v in itertools.product(*params.values())]

            # single_params = {}
            # for key, value in params.items():
            #     if isinstance(value, list):
            #         single_params[key] = value[0]
            #     else:
            #         single_params[key] = value
            for params_combo in param_combinations:
                if isinstance(model_info['model'], XGBClassifier):
                    model = XGBClassifier(objective='binary:logistic',eval_metric = "auc",
                                          early_stopping_rounds = 100 , **params_combo)
                    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)] ,  verbose = False)
                elif isinstance(model_info['model'], LGBMClassifier):
                    import lightgbm as lgb
                    model = LGBMClassifier(objective='binary',
                                           early_stopping_rounds=100,
                                           eval_metric='auc',
                                           verbosity=0,
                                           min_child_samples=20,
                                           min_split_gain=0.1,
                                           **params_combo)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        eval_metric = 'auc',
                        callbacks=[lgb.early_stopping(10)],

                    )
                else:  # CatBoost
                    model = CatBoostClassifier(eval_metric = 'AUC',**params_combo)
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_valid, y_valid),
                        verbose=False,
                        early_stopping_rounds=100
                    )


                y_prob = model.predict_proba(X_valid)[:, 1]
                score = roc_auc_score(y_valid, y_prob)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params_combo
        else:
            # For sklearn models
            cv_splits = [(range(len(X_train)), range(len(X_train), len(X_train) + len(X_valid)))]
            grid_search = GridSearchCV(model_info['model'], params, cv=cv_splits,
                                       scoring=scoring, refit='roc_auc', n_jobs=-1, verbose=1)
            grid_search.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        # Store best model
        self.models[model_name]['best_model'] = best_model

        # Calculate metrics and feature importance
        results = self._calculate_metrics(best_model, X_train, X_valid, y_valid)
        results['best_params'] = best_params

        gc.collect()
        return results

    def _calculate_metrics(self, model, X_train, X_valid, y_valid):
        results = {}

        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            results['feature_importance'] = self.feature_importance.get(model)

        y_pred = model.predict(X_valid)
        y_prob = model.predict_proba(X_valid)[:, 1]

        results.update({
            'validation_roc_auc': roc_auc_score(y_valid, y_prob),
            'validation_average_precision': average_precision_score(y_valid, y_prob),
            'validation_f1': f1_score(y_valid, y_pred)
        })

        return results