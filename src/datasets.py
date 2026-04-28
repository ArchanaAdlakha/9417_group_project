"""
Dataset loading + preprocessing utilities.

This file:
- Loads raw CSV datasets
- Handles missing values
- Drops high-cardinality categorical columns
- Performs train/val/test split
- Applies preprocessing (scaling + encoding)

Output format is always:
(x_train, x_val, x_test, y_train, y_val, y_test, feature_names)
"""

# -------------------- Imports ----------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# global random seed for reproducibility
seed = 42

# -------------------- Feature Type Detection ----------------------
def get_feature_types(x):
    """
    Automatically separate categorical and numerical columns.

    Why:
    - Required for ColumnTransformer
    - Ensures correct preprocessing per feature type
    """
    cat_features = x.select_dtypes(["object", "bool"]).columns.tolist()
    num_features = x.select_dtypes("number").columns.tolist()
    return cat_features, num_features

# -------------------- Dataset Dispatcher --------------------
def load_dataset(name: str):
    """
    Wrapper to load datasets by name.

    Keeps main pipeline clean and avoids repeated if/else logic elsewhere.
    """
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

# -------------------- Feature Cleaning --------------------
def drop_high_cardinality(x, threshold=20):
    """
    Drop categorical columns with too many unique values.

    Why:
    - One-hot encoding high-cardinality features → huge dimensionality
    - Leads to overfitting + slow training

    Example:
    - User IDs, transaction IDs - useless for generalisation
    """
    cat_cols = x.select_dtypes(["object", "bool"]).columns.tolist()
    
    # Identify columns exceeding threshold
    to_drop = [col for col in cat_cols if x[col].nunique() > threshold]
    if to_drop:
        print(f"  Dropping high-cardinality columns: {to_drop}")
    return x.drop(columns=to_drop)

# -------------------- Train/Val/Test Split + Preprocessing --------------------
def split_and_preprocess(x, y, categorical_features, numerical_features, task="classification"):
    """
    Full preprocessing pipeline:
    1. Split data (train / validation / test)
    2. Fit preprocessing on TRAIN only (avoids data leakage)
    3. Transform all splits consistently

    Returns:
    - Numpy arrays ready for ML models
    - Feature names aligned with transformed data
    """

    # 60/20/20 split
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=seed)

    # Numerical: standardised, categorical: one-hot encoded
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # Fit models only on training data
    x_train = preprocessor.fit_transform(x_train)

    # Apply same transformation to val/test
    x_val   = preprocessor.transform(x_val)
    x_test  = preprocessor.transform(x_test)

    # Convert sparse matrices to dense numpy arrays
    if hasattr(x_train, "toarray"):
        x_train = x_train.toarray()
        x_val   = x_val.toarray()
        x_test  = x_test.toarray()

    # Convert to float32 (memory + speed optimisation)
    x_train = np.array(x_train, dtype=np.float32)
    x_val   = np.array(x_val,   dtype=np.float32)
    x_test  = np.array(x_test,  dtype=np.float32)

    y_train = np.array(y_train)
    y_val   = np.array(y_val)
    y_test  = np.array(y_test)

    # ---------------- Feature Name Reconstruction ----------------
    # Important for interpretability (e.g., feature importance plots)
    # ColumnTransformer order:
    #   1. numerical features
    #   2. encoded categorical features
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

# -------------------- Diagnostics --------------------
def inspect_nulls(df, name):
    """
    Print percentage of missing values per column.

    Useful for:
    - deciding threshold for dropping columns
    - debugging datasets
    """
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    print(f"\n{name} — shape: {df.shape}")
    print(null_pct[null_pct > 0].to_string())

def drop_sparse_columns(df, threshold=0.4):
    """
    Drop columns with too many missing values.

    threshold = 0.4 → drop if >40% missing

    Why:
    - High missingness → unreliable features
    - Avoid heavy imputation complexity
    """
    null_frac = df.isnull().sum() / len(df)
    sparse_cols = null_frac[null_frac > threshold].index.tolist()
    return df.drop(columns=sparse_cols)

# -------------------- Dataset Loaders --------------------
def load_diabetes():
    """
    Diabetes readmission dataset (classification).
    Target: readmitted within 30 days.
    """
    df = pd.read_csv(
        "data/Diabetes 130-US Hospitals (classification, 100k, 55, mixed/diabetic_data.csv",
        na_values="?",
        low_memory=False
    )

    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()

    # Subsample for computational efficiency
    df = df.sample(n=15000, random_state=42)
    
    # Binary target
    y = (df["readmitted"] == "<30").astype(int)

    # Remove identifiers (no predictive value)
    x = df.drop(columns=["readmitted", "encounter_id", "patient_nbr"], errors="ignore")
    
    x = drop_high_cardinality(x, threshold=20)

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="classification")

def load_housing():
    """
    Ames Housing dataset (regression).
    Target: SalePrice
    """
    df = pd.read_csv(
        "data/Ames Housing (regression, 1.5k, 79, mixed)/AmesHousing.csv",
        low_memory=False
    )

    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()

    y = df["SalePrice"].values

    # Drop identifiers
    x = df.drop(columns=["SalePrice", "Order", "PID"], errors="ignore")
    
    # Lower threshold due to many categorical variables
    x = drop_high_cardinality(x, threshold=10)  

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="regression")

def load_wine_quality():
    """
    Wine quality dataset (regression).
    Target: quality score
    """
    df = pd.read_csv("data/Wine Quality (Red + White) (regression, 6.5k, 11)/WineQT.csv", low_memory=False)
    
    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()
    
    y = df["quality"].values.astype(float)
    x = df.drop(columns=["quality"])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="regression")

def load_steel_plates_fault():
    """
    Steel faults dataset (classification).
    Multi-class classification via one-hot fault columns.
    """
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
    """
    Online shoppers dataset (classification).
    Target: Revenue (purchase or not)
    """
    df = pd.read_csv("data/Online Shoppers Intention (classification, 12k, 17, mixed)/online_shoppers_intention.csv", low_memory=False)
    
    df = drop_sparse_columns(df, threshold=0.4)
    df = df.dropna()

    y = df["Revenue"].astype(int).values
    x = df.drop(columns=["Revenue"])

    cat_features, num_features = get_feature_types(x)
    return split_and_preprocess(x, y, cat_features, num_features, task="classification")