import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

seed = 42

def get_feature_types(x):
    cat_features = x.select_dtypes(["object", "bool"]).columns.tolist()
    num_features = x.select_dtypes("number").columns.tolist()
    return cat_features, num_features

def load_dataset(name: str):
    """Returns x_train, x_val, x_test, y_train, y_val, y_test"""
    if name == "diabetes": 
        return load_diabetes()
    elif name == "housing":
        return load_housing()
    elif name == "wine":
        return load_wine_quality()
    elif name == "steel":
        return load_steel_plates_fault()
    elif name == "shoppers":
        return load_shoppers()
    else: ValueError(f"Dataset {name} not found")

def split_and_preprocess(x, y, categorical_features, numerical_features, task="classification"):
    # 60/20/20 train/val/test split
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=seed)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    x_train = preprocessor.fit_transform(x_train)
    x_val = preprocessor.transform(x_val)
    x_test = preprocessor.transform(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test, preprocessor

def load_diabetes():
    df = pd.read_csv("data/Diabetes 130-US Hospitals (classification, 100k, 55, mixed/diabetic_data.csv", na_values="?").dropna()

    y = (df["readmitted"] == "<30").astype(int)
    x = df.drop(columns=["readmitted", "encounter_id", "patient_nbr"])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="classification")

def load_housing():
    df = pd.read_csv("data/Ames Housing (regression, 1.5k, 79, mixed)/AmesHousing.csv").dropna()
    
    y = df["SalePrice"].values
    x = df.drop(columns=["SalePrice", "Order", "PID"])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="regression")

def load_wine_quality():
    df = pd.read_csv("data/Wine Quality (Red + White) (regression, 6.5k, 11)/WineQT.csv").dropna()
    
    y = df["quality"].values.astype(float)
    x = df.drop(columns=["quality"])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="regression")


def load_steel_plates_fault():
    df = pd.read_csv("data/Steel Plates Fault (classification, 1.9k, 33)/faults.csv").dropna()
    
    fault_cols = ["Pastry", "Z_Scratch", "K_Scratch", "Stains", 
                  "Dirtiness", "Bumps", "Other_Faults"]
    y = df[fault_cols].values.argmax(axis=1)
    x = df.drop(columns=fault_cols)

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="classification")

def load_shoppers():
    df = pd.read_csv("data/Online Shoppers Intention (classification, 12k, 17, mixed)/online_shoppers_intention.csv").dropna()

    y = df["Revenue"].astype(int).values
    x = df.drop(columns=["Revenue"])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="classification")

