"""
Data preprocessing utilities for NSL-KDD dataset.
Ported from ml/src/data/preprocessing.py for backend inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import joblib
import json
from pathlib import Path

# Column names for NSL-KDD dataset
COLUMN_NAMES = [
    # Basic features (4)
    "duration", "protocol_type", "service", "flag",
    # Content features (13)
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    # Time-based features (9)
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # Host-based features (10)
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    # Labels
    "attack_type", "difficulty_level"
]

# Categorical features
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Numerical features
NUMERICAL_FEATURES = [col for col in COLUMN_NAMES if col not in CATEGORICAL_FEATURES + ["attack_type", "difficulty_level"]]

# Attack type mappings
ATTACK_CATEGORIES = {
    "normal": "normal",
    # DoS attacks
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    # Probe attacks
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L attacks
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l",
    "phf": "r2l", "multihop": "r2l", "warezmaster": "r2l",
    "warezclient": "r2l", "spy": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "snmpguess": "r2l", "snmpgetattack": "r2l", "httptunnel": "r2l",
    "sendmail": "r2l", "named": "r2l",
    # U2R attacks
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r"
}

# Multi-class label mapping
MULTICLASS_LABELS = {
    "normal": 0,
    "dos": 1,
    "probe": 2,
    "r2l": 3,
    "u2r": 4
}

def map_attack_category(attack_type: str) -> str:
    """Map specific attack type to its category."""
    attack_type = str(attack_type).strip().lower()
    return ATTACK_CATEGORIES.get(attack_type, "normal")

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary and multi-class labels from attack types."""
    df = df.copy()
    
    # Map to categories
    if "attack_type" in df.columns:
        df["attack_category"] = df["attack_type"].apply(map_attack_category)
        
        # Binary label: 0=normal, 1=attack
        df["label_binary"] = (df["attack_category"] != "normal").astype(int)
        
        # Multi-class label: 0=normal, 1=dos, 2=probe, 3=r2l, 4=u2r
        df["label_multiclass"] = df["attack_category"].map(MULTICLASS_LABELS)
    
    return df

def encode_categorical_features(df: pd.DataFrame, 
                                encoders: Dict = None,
                                fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """One-hot encode categorical features."""
    df = df.copy()
    
    if fit:
        encoders = {}
        # Store unique values for each categorical feature
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                encoders[col] = sorted(df[col].unique().tolist())
    
    # One-hot encode
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
            
        # Create dummy variables
        dummies = pd.get_dummies(df[col], prefix=col)
        
        # Ensure all expected columns are present
        if encoders and col in encoders:
            for val in encoders[col]:
                col_name = f"{col}_{val}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
                    
            # Keep only expected columns (drop any new unseen values)
            expected_cols = [f"{col}_{val}" for val in encoders[col]]
            # Filter available columns
            available_cols = [c for c in expected_cols if c in dummies.columns]
            # Create missing columns
            missing_cols = [c for c in expected_cols if c not in dummies.columns]
            
            dummies_filtered = dummies[available_cols].copy()
            for c in missing_cols:
                dummies_filtered[c] = 0
                
            # Reorder
            dummies = dummies_filtered[expected_cols]
        
        # Add to dataframe
        df = pd.concat([df, dummies], axis=1)
        
        # Drop original column
        df = df.drop(columns=[col])
    
    return df, encoders

def scale_numerical_features(df: pd.DataFrame,
                            scaler: StandardScaler = None,
                            fit: bool = True) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler."""
    df = df.copy()
    
    # Get numerical columns that exist in dataframe
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    
    if not numerical_cols:
        return df, scaler
        
    if fit:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df, scaler

class NSLKDDPreprocessor:
    """Complete preprocessing pipeline for NSL-KDD dataset."""
    
    def __init__(self):
        self.scaler = None
        self.encoders = None
        self.feature_names = None
        self.fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'NSLKDDPreprocessor':
        """Fit the preprocessor on training data."""
        # Create labels
        df = create_labels(df)
        
        # Encode categorical features
        df, self.encoders = encode_categorical_features(df, fit=True)
        
        # Scale numerical features
        df, self.scaler = scale_numerical_features(df, fit=True)
        
        # Store feature names (exclude labels)
        label_cols = ["attack_type", "difficulty_level", "attack_category", 
                     "label_binary", "label_multiclass"]
        self.feature_names = [col for col in df.columns if col not in label_cols]
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, 
                 include_labels: bool = True) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df = df.copy()
        
        # Create labels if needed
        if include_labels and "attack_type" in df.columns:
            df = create_labels(df)
        
        # Encode categorical features
        df, _ = encode_categorical_features(df, encoders=self.encoders, fit=False)
        
        # Scale numerical features
        df, _ = scale_numerical_features(df, scaler=self.scaler, fit=False)
        
        # Reorder columns to match training
        if not self.feature_names:
            raise ValueError("Feature names not set in preprocessor")
            
        # Add missing features as zeros
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
            
        # Keep only expected features (in correct order)
        feature_cols = df[self.feature_names].copy()
        
        # Ensure all features are float64
        for col in self.feature_names:
            feature_cols[col] = pd.to_numeric(feature_cols[col], errors='coerce').fillna(0.0).astype(np.float64)
        
        # Add labels if available and requested
        if include_labels:
            label_cols = ["attack_type", "attack_category", "label_binary", "label_multiclass"]
            available_labels = [col for col in label_cols if col in df.columns]
            if available_labels:
                return pd.concat([feature_cols, df[available_labels]], axis=1)
        
        return feature_cols
    
    def save(self, path: str):
        """Save preprocessor to disk."""
        if not self.fitted:
            raise ValueError("Cannot save unfitted preprocessor")
            
        save_dict = {
            "encoders": self.encoders,
            "feature_names": self.feature_names
        }
        
        # Save scaler separately
        scaler_path = Path(path).parent / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save rest as JSON
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'NSLKDDPreprocessor':
        """Load preprocessor from disk."""
        preprocessor = cls()
        path_obj = Path(path)
        
        if not path_obj.exists():
             raise FileNotFoundError(f"Preprocessor file not found: {path}")
        
        # Load JSON data
        with open(path, 'r') as f:
            save_dict = json.load(f)
        
        preprocessor.encoders = save_dict["encoders"]
        preprocessor.feature_names = save_dict["feature_names"]
        
        # Load scaler
        scaler_path = path_obj.parent / "scaler.pkl"
        if not scaler_path.exists():
             raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
             
        preprocessor.scaler = joblib.load(scaler_path)
        
        preprocessor.fitted = True
        return preprocessor

