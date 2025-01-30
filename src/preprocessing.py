"""
Preprocessor synchronized with main pipeline.
"""

from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import yaml
import logging

class CTRPreprocessor:
    """Handles data preprocessing for CTR prediction."""

    def __init__(self, config_path: str = 'config/model_config.yaml', target_column: str = 'is_click'):
        """Initialize preprocessor."""
        self.logger = logging.getLogger('CTRPreprocessor')
        self.target_column = target_column

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['preprocessing']

        # Initialize transformers
        self.num_scaler = StandardScaler()
        self.num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        self.label_encoders = {}
        self.scalers = {}
        self.fitted_columns = None

    def validate_columns(self, df: pd.DataFrame, for_training: bool = True) -> None:
        """Validate required columns are present."""
        required_columns = [
            'DateTime',  # Note: matches the case in main.py
            'user_id',
            'product',
            'campaign_id',
            'webpage_id',
            'user_group_id'
        ]

        if for_training:
            required_columns.append(self.target_column)

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}\n"
            error_msg += "Available columns:\n"
            error_msg += "\n".join(f"- {col}" for col in df.columns)
            raise ValueError(error_msg)

    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features."""
        df = df.copy()

        # Ensure DateTime column is datetime type
        df['DateTime'] = pd.to_datetime(df['DateTime'])

        # Create datetime features
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek
        df['is_weekend'] = df['DateTime'].dt.dayofweek.isin([5, 6]).astype(int)
        df['month'] = df['DateTime'].dt.month

        # Drop original DateTime column as it can't be scaled
        df = df.drop('DateTime', axis=1)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df = df.copy()

        # Create basic interactions
        if all(col in df.columns for col in ['user_id', 'product']):
            df['user_product'] = df['user_id'].astype(str) + '_' + df['product'].astype(str)

        if all(col in df.columns for col in ['campaign_id', 'webpage_id']):
            df['campaign_page'] = df['campaign_id'].astype(str) + '_' + df['webpage_id'].astype(str)

        return df

    def _create_aggregation_features(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Create aggregation features."""
        df = df.copy()

        # User level aggregations
        if 'user_id' in df.columns:
            user_aggs = df.groupby('user_id').agg({
                'DateTime': 'count',
                'product': 'nunique',
                'campaign_id': 'nunique'
            }).reset_index()

            user_aggs.columns = ['user_id', 'user_session_count',
                               'user_unique_products', 'user_unique_campaigns']
            df = df.merge(user_aggs, on='user_id', how='left')

        # Product level aggregations
        if 'product' in df.columns:
            product_aggs = df.groupby('product').agg({
                'user_id': 'nunique'
            }).reset_index()

            product_aggs.columns = ['product', 'product_unique_users']
            df = df.merge(product_aggs, on='product', how='left')

        return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unknown categories for test data
                if col in self.label_encoders:
                    # Handle unknown categories by mapping them to a special value
                    df[col] = df[col].astype(str)
                    known_categories = set(self.label_encoders[col].classes_)
                    df.loc[~df[col].isin(known_categories), col] = self.label_encoders[col].classes_[0]
                    df[col] = self.label_encoders[col].transform(df[col])

        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessor and transform training data.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        self.validate_columns(df, for_training=True)

        # Extract target
        y = df[self.target_column].copy()
        X = df.drop(self.target_column, axis=1)

        X = X[~y.isna()]
        y = y[~y.isna()]

        # First, create all features
        X = self._create_datetime_features(X)
        X = self._create_interaction_features(X)
        # X = self._create_aggregation_features(X)

        # Store column names after feature creation but before encoding/scaling
        self.feature_columns = X.columns.tolist()

        # Handle categorical features
        X = self.encode_categorical(X, fit=True)

        # Handle numerical features AFTER all features are created
        num_cols = X.select_dtypes(include=['float64', 'int64', 'int32',np.int64, np.float64]).columns
        if len(num_cols) > 0:
            X[num_cols] = self.num_imputer.fit_transform(X[num_cols])
            X[num_cols] = self.num_scaler.fit_transform(X[num_cols])

        # Store final column order
        self.fitted_columns = X.columns.tolist()

        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform test/validation data.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (transformed features DataFrame, target Series if present)
        """
        self.validate_columns(df, for_training=False)

        # Extract target if present
        if self.target_column in df.columns:
            y = df[self.target_column].copy()
            X = df.drop(self.target_column, axis=1)
        else:
            y = pd.Series(np.nan, index=df.index)
            X = df.copy()

        X = X[~y.isna()]
        y = y[~y.isna()]

        # Create all features first
        X = self._create_datetime_features(X)
        X = self._create_interaction_features(X)
        # X = self._create_aggregation_features(X)

        # Ensure we have all the expected feature columns
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            for col in missing_cols:
                X[col] = 0

        # Remove any extra columns
        extra_cols = set(X.columns) - set(self.feature_columns)
        if extra_cols:
            X = X.drop(columns=extra_cols)

        # Ensure column order matches pre-encoding/scaling order
        X = X[self.feature_columns]

        # Handle categorical features
        X = self.encode_categorical(X, fit=False)

        # Handle numerical features
        num_cols = X.select_dtypes(include=['float64', 'int64', 'int32', 'int32',np.int64, np.float64]).columns
        if len(num_cols) > 0:
            X[num_cols] = self.num_imputer.transform(X[num_cols])
            X[num_cols] = self.num_scaler.transform(X[num_cols])

        # Ensure final column order matches training
        X = X[self.fitted_columns]

        return X, y

