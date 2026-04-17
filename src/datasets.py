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
    else: 
        raise ValueError(f"Dataset {name} not found")

def drop_high_cardinality(x, threshold=20):
    cat_cols = x.select_dtypes(["object", "bool"]).columns.tolist()
    to_drop = [col for col in cat_cols if x[col].nunique() > threshold]
    if to_drop:
        print(f"  Dropping high-cardinality columns: {to_drop}")
    return x.drop(columns=to_drop)

def split_and_preprocess(x, y, categorical_features, numerical_features, task="classification"):
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=seed)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    x_train = preprocessor.fit_transform(x_train)
    x_val   = preprocessor.transform(x_val)
    x_test  = preprocessor.transform(x_test)

    # Convert sparse matrices to dense numpy arrays
    if hasattr(x_train, "toarray"):
        x_train = x_train.toarray()
        x_val   = x_val.toarray()
        x_test  = x_test.toarray()

    x_train = np.array(x_train, dtype=np.float32)
    x_val   = np.array(x_val,   dtype=np.float32)
    x_test  = np.array(x_test,  dtype=np.float32)

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    # Build feature names that match the preprocessed column order:
    #   ColumnTransformer puts "num" columns first, then "cat" columns.
    num_names = list(numerical_features)
    if categorical_features:
        cat_names = (
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(categorical_features)
            .tolist()
        )
    else:
        cat_names = []
    feature_names = num_names + cat_names

    return x_train, x_val, x_test, y_train, y_val, y_test, feature_names

def inspect_nulls(df, name):
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(f"\n{name} — shape: {df.shape}")
    print(null_pct[null_pct > 0].to_string())

def drop_sparse_columns(df, threshold=0.4):
    """Drop columns where more than `threshold` fraction of values are missing."""
    null_frac = df.isnull().sum() / len(df)
    sparse_cols = null_frac[null_frac > threshold].index.tolist()
    return df.drop(columns=sparse_cols)

def load_diabetes():
    df = pd.read_csv(
        "data/Diabetes 130-US Hospitals (classification, 100k, 55, mixed/diabetic_data.csv",
        na_values="?",
        low_memory=False
    )
    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()
    df = df.sample(n=15000, random_state=42)
    
    y = (df["readmitted"] == "<30").astype(int)
    x = df.drop(columns=["readmitted", "encounter_id", "patient_nbr"], errors="ignore")
    x = drop_high_cardinality(x, threshold=20)

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="classification")

def load_housing():
    df = pd.read_csv(
        "data/Ames Housing (regression, 1.5k, 79, mixed)/AmesHousing.csv",
        low_memory=False
    )
    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()

    y = df["SalePrice"].values
    x = df.drop(columns=["SalePrice", "Order", "PID"], errors="ignore")
    x = drop_high_cardinality(x, threshold=10)  

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="regression")

def load_wine_quality():
    df = pd.read_csv("data/Wine Quality (Red + White) (regression, 6.5k, 11)/WineQT.csv", low_memory=False)
    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()
    
    y = df["quality"].values.astype(float)
    x = df.drop(columns=["quality"])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="regression")


def load_steel_plates_fault():
    df = pd.read_csv("data/Steel Plates Fault (classification, 1.9k, 33)/faults.csv", low_memory=False)
    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()

    fault_cols = ["Pastry", "Z_Scratch", "K_Scratch", "Stains", 
                  "Dirtiness", "Bumps", "Other_Faults"]
    fault_cols = [c for c in fault_cols if c in df.columns]

    if fault_cols:
        y = df[fault_cols].values.argmax(axis=1)
        x = df.drop(columns=fault_cols)
    else:
        target_col = df.columns[-1]
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].values)
        x = df.drop(columns=[target_col])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="classification")

def load_shoppers():
    df = pd.read_csv("data/Online Shoppers Intention (classification, 12k, 17, mixed)/online_shoppers_intention.csv", low_memory=False)
    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()

    y = df["Revenue"].astype(int).values
    x = df.drop(columns=["Revenue"])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="classification")