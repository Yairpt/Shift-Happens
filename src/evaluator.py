"""
Comprehensive model evaluation and visualization for CTR prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, classification_report,
    f1_score, log_loss, brier_score_loss, recall_score, precision_score
)
import logging
from pathlib import Path
import yaml
import joblib
from datetime import datetime

from sklearn.model_selection import cross_val_score


class CTREvaluator:
    """
    Handles model evaluation, comparison, and visualization.

    Features:
    - Multiple metric evaluation
    - ROC and PR curves
    - Feature importance analysis
    - Model comparison visualization
    - Calibration plots
    """

    def __init__(self, config_path: str = 'config/model_config.yaml'):
        """
        Initialize evaluator with configuration.

        Args:
            config_path: Path to configuration file
        """
        self._setup_logging()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.evaluation_results = {}

    def _setup_logging(self):
        """Configure logging."""
        self.logger = logging.getLogger('CTREvaluator')
        self.logger.setLevel(logging.INFO)

        # Create handlers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'evaluation_{timestamp}.log')
        ch = logging.StreamHandler()

        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def evaluate_model(self,
                      model_name: str,
                      model,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict:
        """
        Evaluate a single model's performance.

        Args:
            model_name: Name of the model
            model: Trained model instance
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Evaluating model: {model_name}")

        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        results = {
            'model_name': model_name,
            'roc_auc': roc_auc_score(y_test, y_prob),
            'average_precision': average_precision_score(y_test, y_prob),
            'f1': f1_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_prob),
            'brier_score': brier_score_loss(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # Store ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}

        # Store PR curve data
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        results['pr_curve'] = {'precision': precision, 'recall': recall}

        # Store predictions for later analysis
        results['predictions'] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        self.evaluation_results[model_name] = results
        return results

    def plot_roc_curves(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot ROC curves for all evaluated models.

        Args:
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        for model_name, results in self.evaluation_results.items():
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            roc_auc = results['roc_auc']

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def plot_pr_curves(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot Precision-Recall curves for all evaluated models.

        Args:
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        for model_name, results in self.evaluation_results.items():
            precision = results['pr_curve']['precision']
            recall = results['pr_curve']['recall']
            avg_precision = results['average_precision']

            plt.plot(recall, precision,
                    label=f'{model_name} (AP = {avg_precision:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()

    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (15, 5)):
        """
        Plot confusion matrices for all evaluated models.

        Args:
            figsize: Figure size (width, height)
        """
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)

        if n_models == 1:
            axes = [axes]

        for ax, (model_name, results) in zip(axes, self.evaluation_results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                       xticklabels=['No Click', 'Click'],
                       yticklabels=['No Click', 'Click'])
            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model_name: str, top_n: int = 20):
        """
        Plot feature importance for a model if available.

        Args:
            model_name: Name of the model
            top_n: Number of top features to show
        """
        results = self.evaluation_results[model_name]
        if 'feature_importance' not in results:
            self.logger.warning(f"No feature importance available for {model_name}")
            return

        importance_df = results['feature_importance']
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    def plot_metric_comparison(self, metrics: Optional[List[str]] = None):
        """
        Plot comparison of metrics across models.

        Args:
            metrics: List of metrics to compare. If None, uses default metrics.
        """
        if metrics is None:
            metrics = ['roc_auc', 'average_precision', 'f1']

        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            model_metrics = {metric: results[metric] for metric in metrics}
            model_metrics['model'] = model_name
            comparison_data.append(model_metrics)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df_melted = pd.melt(comparison_df,
                                     id_vars=['model'],
                                     var_name='metric',
                                     value_name='score')

        plt.figure(figsize=(12, 6))
        sns.barplot(data=comparison_df_melted, x='model', y='score', hue='metric')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_calibration_curves(self, n_bins: int = 10, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot calibration curves for all models.

        Args:
            n_bins: Number of bins for calibration curve
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        for model_name, results in self.evaluation_results.items():
            y_true = results['predictions']['y_true']
            y_prob = results['predictions']['y_prob']

            # Calculate calibration curve
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_prob, bin_edges) - 1

            bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
            bin_counts = np.bincount(bin_indices, minlength=n_bins)
            bin_means = bin_sums / bin_counts
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            plt.plot(bin_centers, bin_means,
                    marker='o',
                    label=f'{model_name}')

        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curves')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def plot_threshold_impact(self, metric: str = 'f1', figsize: Tuple[int, int] = (10, 6)):
        """
        Plot impact of probability threshold on selected metric.

        Args:
            metric: Metric to evaluate ('f1', 'precision', 'recall')
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        thresholds = np.linspace(0, 1, 100)

        for model_name, results in self.evaluation_results.items():
            y_true = results['predictions']['y_true']
            y_prob = results['predictions']['y_prob']

            metric_values = []
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                if metric == 'f1':
                    score = f1_score(y_true, y_pred)
                elif metric == 'precision':
                    score = precision_score(y_true, y_pred)
                elif metric == 'recall':
                    score = recall_score(y_true, y_pred)
                metric_values.append(score)

            plt.plot(thresholds, metric_values, label=model_name)

        plt.xlabel('Classification Threshold')
        plt.ylabel(f'{metric.title()} Score')
        plt.title(f'Impact of Classification Threshold on {metric.title()}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_learning_curves(self, X_train: pd.DataFrame, y_train: pd.Series,
                           cv: int = 5, scoring: str = 'roc_auc',
                           train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
                           figsize: Tuple[int, int] = (10, 6)):
        """
        Plot learning curves for all models.

        Args:
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
            scoring: Scoring metric
            train_sizes: Array of training size proportions
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        for model_name, results in self.evaluation_results.items():
            if not hasattr(results['model'], 'fit'):
                continue

            train_scores = []
            val_scores = []

            for size in train_sizes:
                train_idx = np.random.choice(len(X_train),
                                           size=int(size * len(X_train)),
                                           replace=False)
                X_subset = X_train.iloc[train_idx]
                y_subset = y_train.iloc[train_idx]

                scores = cross_val_score(results['model'], X_subset, y_subset,
                                       scoring=scoring, cv=cv)

                train_scores.append(np.mean(scores))
                val_scores.append(np.std(scores))

            plt.errorbar(train_sizes, train_scores, yerr=val_scores,
                        label=model_name, capsize=5)

        plt.xlabel('Training Set Size')
        plt.ylabel(f'{scoring} Score')
        plt.title('Learning Curves')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    def plot_error_analysis(self, X_test: pd.DataFrame, top_n: int = 10,
                           figsize: Tuple[int, int] = (15, 6)):
        """
        Plot analysis of prediction errors by feature values.

        Args:
            X_test: Test features
            top_n: Number of top features to analyze
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for model_name, results in self.evaluation_results.items():
            y_true = results['predictions']['y_true']
            y_pred = results['predictions']['y_pred']

            # Identify errors
            errors = y_true != y_pred

            # False Positives
            fp_mask = (y_pred == 1) & (y_true == 0)
            fp_data = X_test[fp_mask]

            # False Negatives
            fn_mask = (y_pred == 0) & (y_true == 1)
            fn_data = X_test[fn_mask]

            # Analyze numerical features
            num_features = X_test.select_dtypes(include=['float64', 'int64']).columns

            for i, feature in enumerate(num_features[:top_n]):
                # Plot FP distribution
                sns.kdeplot(data=fp_data[feature], ax=axes[0],
                          label=f'{model_name} - {feature}')

                # Plot FN distribution
                sns.kdeplot(data=fn_data[feature], ax=axes[1],
                          label=f'{model_name} - {feature}')

        axes[0].set_title('False Positive Error Distribution')
        axes[1].set_title('False Negative Error Distribution')
        for ax in axes:
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_probability_distributions(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot probability distribution for each class by model.

        Args:
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)

        for model_name, results in self.evaluation_results.items():
            y_true = results['predictions']['y_true']
            y_prob = results['predictions']['y_prob']

            # Plot distributions for each class
            for label in [0, 1]:
                mask = y_true == label
                sns.kdeplot(y_prob[mask],
                           label=f'{model_name} (Class {label})')

        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distributions by Class and Model')
        plt.legend()
        plt.grid(True)
        plt.show()

    def generate_report(self, output_path: Union[str, Path]):
        """
        Generate comprehensive evaluation report.

        Args:
            output_path: Path to save the report
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {}
        }

        for model_name, results in self.evaluation_results.items():
            model_report = {
                'metrics': {
                    'roc_auc': float(results['roc_auc']),
                    'average_precision': float(results['average_precision']),
                    'f1': float(results['f1']),
                    'log_loss': float(results['log_loss']),
                    'brier_score': float(results['brier_score'])
                },
                'classification_report': results['classification_report']
            }
            report['models'][model_name] = model_report

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        self.logger.info(f"Evaluation report saved to {output_path}")