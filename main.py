"""
Main pipeline script for CTR prediction project.
"""
import os

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
import warnings

warnings.filterwarnings('ignore')

# Import our modules
from src.preprocessing import CTRPreprocessor
from src.model_trainer import CTRModelTrainer
from src.evaluator import CTREvaluator
from src.utils import DataUtils, LoggingUtils, ModelUtils


class CTRPipeline:
    """Main pipeline for CTR prediction project."""

    def __init__(self, config_path: str = 'config/model_config.yaml'):
        """Initialize pipeline with configuration."""
        self.logger = LoggingUtils.setup_logger('CTRPipeline', 'logs/pipeline.log')
        self.config_path = config_path
        self.output_dir = Path('output')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, data_path: str, target_column: str):
        """
        Run the complete pipeline.

        Args:
            data_path: Path to input data file
        """
        self.logger.info("Starting CTR prediction pipeline")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        try:
            # Load and check data
            self.logger.info("Loading data...")
            data_utils = DataUtils()
            df = data_utils.load_data(data_path)

            # Log data quality report
            quality_report = data_utils.check_data_quality(df)
            self.logger.info(f"Data quality report: {quality_report}")

            # Split data
            self.logger.info("Splitting data...")
            train_df, valid_df, test_df = data_utils.split_time_series(
                df, 'DateTime', train_ratio=0.7, valid_ratio=0.15
            )

            # Preprocess data
            self.logger.info("Preprocessing data...")

            # Print column names for debugging
            self.logger.info("Available columns in dataset:")
            for col in df.columns:
                self.logger.info(f"  - {col}")

            preprocessor = CTRPreprocessor(self.config_path, target_column=target_column)

            try:
                preprocessor.validate_columns(df)
            except ValueError as e:
                self.logger.error(f"Column validation failed: {str(e)}")
                print("\nPlease ensure your dataset contains the following columns:")
                print("  - datetime (timestamp of the event)")
                print(f"  - {target_column} (target variable, 0 or 1)")
                print("  - user_id (unique identifier for users)")
                print("  - product (product identifier)")
                print("  - campaign_id (campaign identifier)")
                print("  - webpage_id (webpage identifier)")
                print("  - user_group_id (user group identifier)")
                print("\nYour dataset contains the following columns:")
                print("\n".join(f"  - {col}" for col in df.columns))
                raise

            X_train, y_train = preprocessor.fit_transform(train_df)
            X_valid, y_valid = preprocessor.transform(valid_df)
            X_test, y_test = preprocessor.transform(test_df)

            # Train and evaluate models
            self.logger.info("Training models...")
            trainer = CTRModelTrainer(self.config_path)
            evaluator = CTREvaluator(self.config_path)

            # Store results for comparison
            model_results = {}
            best_score = -np.inf
            best_model_name = None

            # Train each model
            for model_name in trainer.models.keys():
                self.logger.info(f"Training {model_name}...")

                # Train model
                results = trainer.train_evaluate_model(
                    X_train, y_train,
                    X_valid, y_valid,
                    model_name
                )

                # Evaluate on test set
                evaluation = evaluator.evaluate_model(
                    model_name,
                    trainer.models[model_name]['best_model'],
                    X_test,
                    y_test
                )

                model_results[model_name] = {
                    'training_results': results,
                    'test_evaluation': evaluation
                }

                # Track best model
                if evaluation['roc_auc'] > best_score:
                    best_score = evaluation['roc_auc']
                    best_model_name = model_name

            # Generate visualizations
            self.logger.info("Generating visualizations...")
            evaluator.plot_roc_curves()
            evaluator.plot_pr_curves()
            evaluator.plot_confusion_matrices()
            evaluator.plot_metric_comparison()
            evaluator.plot_calibration_curves()

            # Save best model
            self.logger.info(f"Best model: {best_model_name}")
            best_model = trainer.models[best_model_name]['best_model']
            ModelUtils.save_model(
                best_model,
                self.output_dir / 'models',
                f'best_model_{timestamp}'
            )

            # Generate summary report
            summary = {
                'timestamp': timestamp,
                'best_model': {
                    'name': best_model_name,
                    'roc_auc': float(best_score),
                    'parameters': best_model.get_params()
                },
                'model_comparison': {
                    name: {
                        'roc_auc': float(results['test_evaluation']['roc_auc']),
                        'f1': float(results['test_evaluation']['f1']),
                        'best_params': results['training_results']['best_params']
                    }
                    for name, results in model_results.items()
                }
            }

            # Save summary
            with open(self.output_dir / f'summary_{timestamp}.yaml', 'w') as f:
                yaml.dump(summary, f)

            # Print summary
            print("\n=== Pipeline Results ===")
            print(f"\nBest Model: {best_model_name}")
            print(f"ROC-AUC Score: {best_score:.4f}")
            print("\nBest Parameters:")
            for name, results in model_results.items():
                print(f"\n{name}:")
                print(f"Best params: {results['training_results']['best_params']}")
                print(f"ROC-AUC: {results['test_evaluation']['roc_auc']:.4f}")

            self.logger.info("Pipeline completed successfully")
            return summary

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    # Run pipeline
    pipeline = CTRPipeline()
    # summary = pipeline.run_pipeline('data/train_dataset_full.csv', "is_click")
    summary = pipeline.run_pipeline('data/train_dataset_partial.csv', "is_click")