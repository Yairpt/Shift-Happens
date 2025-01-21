import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def parse(csv_path):
    print(f"location of the file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(df.head())
    df.drop_duplicates(inplace=True)
    return df

def imputation(df: pd.DataFrame) -> pd.DataFrame:
    num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    categor_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    num_column = []
    for column in df.columns:
        if df[column].dtype == np.float64:
            df[column] = num_imputer.fit_transform(df[[column]]).ravel()
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = (
                df[column].fillna(df[column].mode()[0]))
        else:
            df[column] = categor_imputer.fit_transform(df[[column]]).ravel()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("Loading data...")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    print(df.dtypes)
    print(df.isna().sum())
    imputation(df)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str)
    args = parser.parse_args()

    df = parse(args.csv_path)
    preprocess(df)