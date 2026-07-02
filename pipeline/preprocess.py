import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

def make_rf_pipeline(
    numeric_features,
    categorical_features,
    categorical_strategy="unknown"
):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    if categorical_strategy == "unknown":
        categorical_transformer = Pipeline(steps=[
            # Step 1: allow string values like "Unknown"
            ("to_object", FunctionTransformer(lambda x: x.astype(object), validate=False)),
            
            # Step 2: fill missing values with "Unknown"
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            
            # Step 3: make all categories strings: 0.0 -> "0.0", 1.0 -> "1.0", Unknown -> "Unknown"
            ("to_string", FunctionTransformer(lambda x: x.astype(str), validate=False)),
            
            # Step 4: one-hot encode
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

    elif categorical_strategy == "most_frequent":
        categorical_transformer = Pipeline(steps=[
            ("to_object", FunctionTransformer(lambda x: x.astype(object), validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_string", FunctionTransformer(lambda x: x.astype(str), validate=False)),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

    else:
        raise ValueError("categorical_strategy must be 'unknown' or 'most_frequent'")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return Pipeline(steps=[
        ("preprocess", preprocessor)
    ])

def insertSegmentData(df):
    # Insert segment data into the dataframe
    # This is a placeholder for the actual implementation
    df['Segment'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    return df

def tStageClean(df):
    # Clean T stage data in the dataframe
    numericFeatures = ["Age"]
    categoricalFeatures = ["Gender", "CenterID", "Tobacco Consumption", "Alcohol Consumption", "HPV Status"]

def nStageClean(df):
    # Clean N stage data in the dataframe
    # This is a placeholder for the actual implementation
    df['N_Stage'] = df['N_Stage'].str.upper()
    return df

def rfsClean(df):
    # Clean RFS data in the dataframe
    # This is a placeholder for the actual implementation
    df['RFS'] = df['RFS'].fillna('Unknown')
    return df

def main():
    rawDF = pd.read_csv('../Data/HECKTOR_2026_training_data.csv')

    segmentDF = insertSegmentData(rawDF)

    tStageDF = tStageClean(segmentDF)
    nStageDF = nStageClean(tStageDF)
    rfsDF = rfsClean(nStageDF)
