"""
Utility functions for CTR prediction pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import yaml
import joblib
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split


class DataUtils:
    """Utilities for data handling and preprocessing."""

    @staticmethod
    def load_data(file_path: Union[str, Path], date_columns: List[str] = None) -> pd.DataFrame:
        """
        Load data from various file formats and handle datetime conversion.

        Args:
            file_path: Path to data file
            date_columns: List of column names to be converted to datetime

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)

        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Convert date columns
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to datetime. Error: {str(e)}")

        return df

    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate missing value statistics.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing value statistics
        """
        missing_stats = pd.DataFrame({
            'missing_count': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2)
        }).reset_index()
        missing_stats.columns = ['feature', 'missing_count', 'missing_percentage']
        return missing_stats[missing_stats['missing_count'] > 0].sort_values(
            'missing_percentage', ascending=False
        )

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict:
        """
        Perform basic data quality checks.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2,  # MB
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': DataUtils.check_missing_values(df).to_dict('records')
        }
        return quality_report

    @staticmethod
    def split_time_series(df: pd.DataFrame,
                          date_column: str,
                          train_ratio: float = 0.7,
                          valid_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data preserving temporal order.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            train_ratio: Proportion of data for training
            valid_ratio: Proportion of data for validation

        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])

        df = df.sort_values(date_column)

        train_end = int(len(df) * train_ratio)
        valid_end = int(len(df) * (train_ratio + valid_ratio))

        train_df = df.iloc[:train_end]
        valid_df = df.iloc[train_end:valid_end]
        test_df = df.iloc[valid_end:]

        return train_df, valid_df, test_df


class ModelUtils:
    """Utilities for model operations."""

    @staticmethod
    def save_model(model, path: Union[str, Path], model_name: str):
        """
        Save trained model and metadata.

        Args:
            model: Trained model instance
            path: Directory path
            model_name: Name of the model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, path / f"{model_name}.joblib")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': str(type(model).__name__),
            'parameters': model.get_params() if hasattr(model, 'get_params') else {}
        }

        with open(path / f"{model_name}_metadata.yaml", 'w') as f:
            yaml.dump(metadata, f)

    @staticmethod
    def load_model(path: Union[str, Path], model_name: str):
        """
        Load saved model and metadata.

        Args:
            path: Directory path
            model_name: Name of the model

        Returns:
            Loaded model instance
        """
        path = Path(path)

        # Load model
        model = joblib.load(path / f"{model_name}.joblib")

        # Load metadata
        with open(path / f"{model_name}_metadata.yaml", 'r') as f:
            metadata = yaml.safe_load(f)

        return model, metadata


class MetricUtils:
    """Utilities for metric calculation and analysis."""

    @staticmethod
    def calculate_ctr(clicks: int, impressions: int) -> float:
        """
        Calculate Click-Through Rate.

        Args:
            clicks: Number of clicks
            impressions: Number of impressions

        Returns:
            CTR value
        """
        return (clicks / impressions * 100) if impressions > 0 else 0

    @staticmethod
    def calculate_confidence_interval(ctr: float,
                                      impressions: int,
                                      confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for CTR.

        Args:
            ctr: Click-Through Rate value
            impressions: Number of impressions
            confidence: Confidence level

        Returns:
            Tuple of (lower bound, upper bound)
        """
        from scipy import stats

        ctr = ctr / 100  # Convert to proportion
        standard_error = np.sqrt((ctr * (1 - ctr)) / impressions)
        z_score = stats.norm.ppf((1 + confidence) / 2)

        margin_of_error = z_score * standard_error
        lower_bound = max(0, (ctr - margin_of_error)) * 100
        upper_bound = min(1, (ctr + margin_of_error)) * 100

        return lower_bound, upper_bound


class LoggingUtils:
    """Utilities for logging and monitoring."""

    @staticmethod
    def setup_logger(name: str,
                     log_file: Optional[str] = None,
                     level: int = logging.INFO) -> logging.Logger:
        """
        Set up logger with formatting.

        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Logging level

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def log_experiment(experiment_name: str,
                       parameters: Dict,
                       metrics: Dict,
                       artifacts_path: Optional[Union[str, Path]] = None):
        """
        Log experiment parameters and results.

        Args:
            experiment_name: Name of the experiment
            parameters: Dictionary of parameters
            metrics: Dictionary of metrics
            artifacts_path: Path to save artifacts (optional)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'parameters': parameters,
            'metrics': metrics
        }

        if artifacts_path:
            artifacts_path = Path(artifacts_path)
            artifacts_path.mkdir(parents=True, exist_ok=True)

            with open(artifacts_path / f"{experiment_name}_{timestamp}.yaml", 'w') as f:
                yaml.dump(experiment_data, f)