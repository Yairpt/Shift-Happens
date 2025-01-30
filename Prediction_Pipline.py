import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
import yaml
import json


class CTRPredictor:
    """
    A class to handle CTR prediction pipeline including preprocessing, model training,
    and evaluation.
    """

    def __init__(self, config_path='model_config.yaml'):
        """
        Initialize the predictor with configuration.

        Parameters:
        -----------
        config_path : str
            Path to the YAML configuration file
        """
        self.models = {}
        self.results = {}
        self.feature_importances = {}
        self.load_config(config_path)

    def load_config(self, config_path):
        """Load model configurations from YAML file"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def preprocess_data(self, df):
        """
        Preprocess the input dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with raw data

        Returns:
        --------
        X : numpy.ndarray
            Processed feature matrix
        y : numpy.ndarray
            Target variable array
        """
        # Create copy to avoid modifying original data
        data = df.copy()

        # Handle categorical variables
        categorical_features = ['product', 'product_category_1', 'product_category_2',
                                'gender', 'age_level', 'user_depth']

        label_encoders = {}
        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                data[feature] = le.fit_transform(data[feature].astype(str))
                label_encoders[feature] = le

        # Handle datetime
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['month'] = data['datetime'].dt.month

        # Drop unnecessary columns
        columns_to_drop = ['datetime', 'session_id', 'user_id', 'campaign_id', 'webpage_id']
        X = data.drop(columns_to_drop + ['is_click'], axis=1)
        y = data['is_click']

        # Scale numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y

    def train_and_evaluate(self, X, y):
        """
        Train models and evaluate their performance.

        Parameters:
        -----------
        X : numpy.ndarray
            Processed feature matrix
        y : numpy.ndarray
            Target variable array
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model_config in self.config['models'].items():
            print(f"\nTraining {model_name}...")

            # Initialize model with base parameters
            model_class = eval(model_config['class'])
            model = model_class(**model_config['base_params'])

            # Perform GridSearch with cross-validation
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=model_config['grid_params'],
                cv=kf,
                scoring=['roc_auc', 'average_precision', 'f1'],
                refit='roc_auc'
            )

            grid_search.fit(X, y)

            # Store results
            self.models[model_name] = grid_search.best_estimator_
            self.results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }

            # Store feature importances for applicable models
            if hasattr(grid_search.best_estimator_, 'feature_importances_'):
                self.feature_importances[model_name] = grid_search.best_estimator_.feature_importances_

    def visualize_results(self):
        """Create visualizations for model comparison and evaluation"""
        # Model Performance Comparison
        plt.figure(figsize=(12, 6))
        scores = {model: self.results[model]['best_score'] for model in self.models.keys()}
        plt.bar(scores.keys(), scores.values())
        plt.title('Model Performance Comparison (ROC-AUC)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Feature Importance Plot (for applicable models)
        for model_name, importances in self.feature_importances.items():
            plt.figure(figsize=(12, 6))
            features = range(len(importances))
            plt.bar(features, importances)
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.show()

        # Cross-validation Results
        plt.figure(figsize=(12, 6))
        for model_name in self.models.keys():
            cv_results = self.results[model_name]['cv_results']
            plt.plot(cv_results['mean_test_roc_auc'], label=model_name)
        plt.title('Cross-validation Results - ROC-AUC')
        plt.xlabel('Iteration')
        plt.ylabel('ROC-AUC Score')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_results(self, output_path='results.json'):
        """Save evaluation results to a JSON file"""
        results_dict = {}
        for model_name in self.models.keys():
            results_dict[model_name] = {
                'best_params': self.results[model_name]['best_params'],
                'best_score': float(self.results[model_name]['best_score'])
            }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=4)