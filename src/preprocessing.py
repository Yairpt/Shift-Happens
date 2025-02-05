"""
Preprocessor synchronized with main pipeline.
"""

from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
            'product_category_1',
            'webpage_id',
            'user_group_id',
            'gender',
            'age_level',
            'user_depth',
            'city_development_index',
            'is_click',
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
        # User interactions
        df['user_campaign'] = df['user_id'].astype(str) + '_' + df['campaign_id'].astype(str)
        df['user_product'] = df['user_id'].astype(str) + '_' + df['product'].astype(str)
        df['user_webpage'] = df['user_id'].astype(str) + '_' + df['webpage_id'].astype(str)

        # Product interactions
        df['product_campaign'] = df['product'].astype(str) + '_' + df['campaign_id'].astype(str)
        df['product_webpage'] = df['product'].astype(str) + '_' + df['webpage_id'].astype(str)
        df['product_categories'] = df['product_category_1'].astype(str) + '_' + df['product_category_2'].astype(str)

        # Time interactions
        df['hour_weekday'] = df['hour'].astype(str) + '_' + df['day_of_week'].astype(str)
        df['hour_weekend'] = df['hour'].astype(str) + '_' + df['is_weekend'].astype(str)
        df['hour_day'] = df['hour'].astype(str) + '_' + df['day_of_week'].astype(str)
        # Location interactions
        df['user_city'] = df['user_id'].astype(str) + '_' + df['city_development_index'].astype(str)

        # Category interaction
        df['category_combined'] = df['product_category_1'].astype(str) + '_' + df['product_category_2'].astype(str)
        return df

    def _create_aggregation_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Create aggregation features."""
        df = df.copy()
        df['is_click'] = y

        # User level aggregations
        user_features = df.groupby('user_id').agg({
            'is_click': ['mean', 'count'],
            'product': 'nunique',
            'campaign_id': 'nunique',
            'webpage_id': 'nunique'
        })

        # Flatten column names
        user_features.columns = [f'user_{col[0]}_{col[1]}' for col in user_features.columns]

        # Merge user features
        df = df.merge(user_features, left_on='user_id', right_index=True, how='left')


        # Hour-based CTR
        hourly_ctr = df.groupby('hour')['is_click'].mean()
        df['hour_ctr'] = df['hour'].map(hourly_ctr)

        # Product CTR
        product_ctr = df.groupby('product')['is_click'].mean()
        df['product_ctr'] = df['product'].map(product_ctr)

        # Aggregate by city, age, product
        city_age_product_agg = df.groupby(["city_development_index", "age_level", "product"]).agg({
            "is_click": ["sum", "count"],
            "campaign_id": "nunique",
            "webpage_id": "nunique"
        }).reset_index()

        # Rename columns after aggregation
        city_age_product_agg.columns = ["city_development_index", "age_level", "product",
                                        "click_sum_city_age_prod", "click_count_city_age_prod",
                                        "unique_campaigns_city_age_prod", "unique_webpages_city_age_prod"]

        # Merge into train & test dataset
        df = df.merge(city_age_product_agg, on=["city_development_index", "age_level", "product"], how="left")

        df = df.drop('is_click', axis=1)
        df.to_csv('data/processed_data_after_feature_engineering.csv', index=False)
        return df
    @staticmethod
    def impute_by_most_frequent_value(df: pd.DataFrame, reference_col: str, target_col: str) -> pd.DataFrame:
        """
        Impute missing values in target_col based on most frequent values associated with reference_col.

        Args:
            df: Input dataframe
            reference_col: Column used as reference (e.g., campaign_id)
            target_col: Column with missing values to impute (e.g., webpage_id)
        """
        df = df.copy()

        # Get most frequent target values for each reference value
        most_frequent_values = df.groupby(reference_col)[target_col].agg(
            lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
        )

        # Create mapping dictionary
        mapping_dict = most_frequent_values.to_dict()

        # Fill missing values
        mask = df[target_col].isna()
        df.loc[mask, target_col] = df.loc[mask, reference_col].map(mapping_dict)

        return df

    def handle_missing_values(self, df: pd.DataFrame, target_col: pd.Series) -> pd.DataFrame:
        """
        Handles missing values in the dataset following specific rules for each column:

        Pre-processing:
        - Drops rows with >3 missing columns
        - Drops rows with missing target values if target column exists

        Column-specific rules:
        1. product_category_2: Fill with 0
        2. session_id: Impute from user_id, then fill remaining with 0
        3. DateTime: Fill with 12:00 noon
        4. user_id: Impute from session_id, then 0
        5. product: Impute from product_category_1, then user_id, then 'U'
        6. campaign_id: Impute from webpage_id, then user_id, then 99
        7. webpage_id: Impute from campaign_id, then user_id, then 99
        8. product_category_1: Impute from product, then campaign_id, then 99
        9. product_category_2: Fill with 99
        10. user_group: Impute from user_id, then age_level, then 99
        11. age_level: Impute from user_id, then user_group_id, then 99
        12. gender: Impute from user_id, then 'Unknown'
        13. age_level: Impute from user_id, then user_group_id, then 99
        14. user_depth: Impute from user_id, then user_group_id, then 99
        15. city_development_index: Impute from user_id, then user_group_id, then 99
        16. var1: Fill with 0
        """
        df = df.copy()

        # Pre-processing - drop target NaN first
        df["is_click"] = target_col
        df = df[~target_col.isnull()]

        # Pre-processing
        df = df[df.isnull().sum(axis=1) <= 3]

        # Apply imputation rules
        df['product_category_2'].fillna(0, inplace=True)

        df = self.impute_by_most_frequent_value(df, 'user_id', 'session_id')
        df['session_id'].fillna(0, inplace=True)

        df['DateTime'].fillna(pd.Timestamp('12:00:00'), inplace=True)

        df = self.impute_by_most_frequent_value(df, 'session_id', 'user_id')
        df['user_id'].fillna(0, inplace=True)

        df = self.impute_by_most_frequent_value(df, 'product_category_1', 'product')
        df = self.impute_by_most_frequent_value(df, 'user_id', 'product')
        df['product'].fillna('U', inplace=True)

        df = self.impute_by_most_frequent_value(df, 'webpage_id', 'campaign_id')
        df = self.impute_by_most_frequent_value(df, 'user_id', 'campaign_id')
        df['campaign_id'].fillna(99, inplace=True)

        df = self.impute_by_most_frequent_value(df, 'campaign_id', 'webpage_id')
        df = self.impute_by_most_frequent_value(df, 'user_id', 'webpage_id')
        df['webpage_id'].fillna(99, inplace=True)

        df = self.impute_by_most_frequent_value(df, 'product', 'product_category_1')
        df = self.impute_by_most_frequent_value(df, 'campaign_id', 'product_category_1')
        df['product'].fillna('U', inplace=True)
        df = self.impute_by_most_frequent_value(df,  'product_category_1', 'product')
        df['product_category_1'].fillna(99, inplace=True)

        df['product_category_2'].fillna(99, inplace=True)

        for col in ['user_group_id', 'age_level', 'user_depth', 'city_development_index']:
            df = self.impute_by_most_frequent_value(df, 'user_id', col)
            df = self.impute_by_most_frequent_value(df, 'user_group_id', col)
            df[col].fillna(99, inplace=True)

        df = self.impute_by_most_frequent_value(df, 'user_id', 'gender')
        df['gender'].fillna('Unknown', inplace=True)

        df['var_1'].fillna(0, inplace=True)

        # Drop any remaining rows with NaN values
        df.dropna(inplace=True)
        df.to_csv('data/processed_data_after_imputation.csv', index=False)

        y = df['is_click'].copy()
        df = df.drop('is_click', axis=1)
        return df, y

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

        # Drop rows with missing target values
        X = X[~y.isna()]
        y = y[~y.isna()]

        # First, create all features
        X, y = self.handle_missing_values(X, y)
        X = self._create_datetime_features(X)
        X = self._create_interaction_features(X)
        X = self._create_aggregation_features(X, y)

        # Store column names after feature creation but before encoding/scaling
        self.feature_columns = X.columns.tolist()

        # Handle categorical features
        X = self.encode_categorical(X, fit=True)

        # Probabably unnecessary to scale the categorical features
        # Handle numerical features AFTER all features are created
        num_cols = X.select_dtypes(include=['float64', 'int64', 'int32',np.int64, np.float64]).columns
        if len(num_cols) > 0:
            X[num_cols] = self.num_imputer.fit_transform(X[num_cols])
        #     X[num_cols] = self.num_scaler.fit_transform(X[num_cols])

        # Store final column order
        self.fitted_columns = X.columns.tolist()
        X.to_csv('data/processed_data_final.csv', index=False)
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
        X, y = self.handle_missing_values(X, y)
        X = self._create_datetime_features(X)
        X = self._create_interaction_features(X)
        X = self._create_aggregation_features(X, y)

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
            # X[num_cols] = self.num_scaler.transform(X[num_cols])

        # Ensure final column order matches training
        X = X[self.fitted_columns]

        return X, y

    def visualize_aggregated_features(self):

        age_sex_product_agg = None
        city_age_product_agg = None

        # Set style for better readability
        sns.set_style("whitegrid")
        palette = sns.color_palette("coolwarm", as_cmap=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        #  Total Clicks by Age & Gender vs Product
        sns.barplot(x="age_level", y="click_sum_age_sex_prod", hue="gender",
                    data=age_sex_product_agg, ax=axes[0, 0], palette="coolwarm")
        axes[0, 0].set_title("Total Clicks by Age & Gender vs Product", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("Age Level", fontsize=12)
        axes[0, 0].set_ylabel("Total Clicks", fontsize=12)
        axes[0, 0].legend(title="Gender")

        #  Total Clicks by City Development Index, Age & Product
        sns.barplot(x="city_development_index", y="click_sum_city_age_prod", hue="age_level",
                    data=city_age_product_agg, ax=axes[0, 1], palette="viridis")
        axes[0, 1].set_title("Total Clicks by City, Age & Product", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("City Development Index", fontsize=12)
        axes[0, 1].set_ylabel("Total Clicks", fontsize=12)
        axes[0, 1].legend(title="Age Level")

        # ️⃣ Unique Campaigns by Age & Gender vs Product
        sns.barplot(x="age_level", y="unique_campaigns_age_sex_prod", hue="gender",
                    data=age_sex_product_agg, ax=axes[1, 0], palette="Set2")
        axes[1, 0].set_title("Unique Campaigns by Age & Gender vs Product", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("Age Level", fontsize=12)
        axes[1, 0].set_ylabel("Unique Campaigns", fontsize=12)
        axes[1, 0].legend(title="Gender")

        #  Unique Webpages by City Development Index, Age & Product
        sns.barplot(x="city_development_index", y="unique_webpages_city_age_prod", hue="age_level",
                    data=city_age_product_agg, ax=axes[1, 1], palette="magma")
        axes[1, 1].set_title("Unique Webpages by City, Age & Product", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("City Development Index", fontsize=12)
        axes[1, 1].set_ylabel("Unique Webpages", fontsize=12)
        axes[1, 1].legend(title="Age Level")

        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()

