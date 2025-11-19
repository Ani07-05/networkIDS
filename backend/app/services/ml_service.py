"""
ML inference service for loading and running ONNX models.
"""

import onnxruntime as ort
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from functools import lru_cache
import os

from app.config import get_settings
from app.services.preprocessing import NSLKDDPreprocessor

settings = get_settings()


class MLInferenceService:
    """Service for ML model inference using ONNX Runtime."""
    
    def __init__(self):
        """Initialize ML inference service."""
        self.binary_session = None
        self.multiclass_session = None
        self.preprocessor = None
        self.attack_types = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        self._load_models()
        self._load_preprocessor()
    
    def _resolve_path(self, path_str: str) -> Path:
        """
        Resolve path relative to current working directory or project root.
        Handles running from backend/ vs project root.
        """
        path = Path(path_str)
        if path.exists():
            return path
            
        # Try going up one level (if running from backend/)
        parent_path = Path("..") / path_str
        if parent_path.exists():
            return parent_path
            
        return path
    
    def _load_models(self):
        """Load ONNX models."""
        try:
            # Load binary classification model
            binary_path = self._resolve_path(settings.BINARY_MODEL_PATH)
            if binary_path.exists():
                self.binary_session = ort.InferenceSession(
                    str(binary_path),
                    providers=['CPUExecutionProvider']
                )
                print(f"[OK] Loaded binary model from {binary_path}")
            else:
                print(f"[WARNING] Binary model not found at {binary_path}")
            
            # Load multiclass classification model
            multiclass_path = self._resolve_path(settings.MULTICLASS_MODEL_PATH)
            if multiclass_path.exists():
                self.multiclass_session = ort.InferenceSession(
                    str(multiclass_path),
                    providers=['CPUExecutionProvider']
                )
                print(f"[OK] Loaded multiclass model from {multiclass_path}")
            else:
                print(f"[WARNING] Multiclass model not found at {multiclass_path}")
                
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
    
    def _load_preprocessor(self):
        """Load preprocessing configuration."""
        try:
            preprocessor_path = self._resolve_path(settings.PREPROCESSOR_PATH)
            if preprocessor_path.exists():
                self.preprocessor = NSLKDDPreprocessor.load(str(preprocessor_path))
                print(f"[OK] Loaded preprocessor from {preprocessor_path}")
            else:
                print(f"[WARNING] Preprocessor not found at {preprocessor_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load preprocessor: {e}")
    
    def preprocess_features(self, features: Dict) -> np.ndarray:
        """
        Preprocess raw features for model input.
        
        Args:
            features: Dictionary of network traffic features
            
        Returns:
            Preprocessed feature array
        """
        if not self.preprocessor:
            raise ValueError("Preprocessor not loaded")
        
        # Convert single feature dict to DataFrame
        df = pd.DataFrame([features])
        
        # Transform using loaded preprocessor
        processed_df = self.preprocessor.transform(df, include_labels=False)
        
        # Convert to numpy array (float32 for ONNX)
        feature_array = processed_df.values.astype(np.float32)
        
        return feature_array
    
    def preprocess_batch(self, features_list: List[Dict]) -> np.ndarray:
        """
        Preprocess a batch of features.
        """
        if not self.preprocessor:
            raise ValueError("Preprocessor not loaded")
            
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(features_list)
        
        # Transform
        processed_df = self.preprocessor.transform(df, include_labels=False)
        
        return processed_df.values.astype(np.float32)
    
    def predict_binary(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Make binary classification prediction (Normal vs Attack).
        
        Args:
            features: Preprocessed feature array (shape: [1, n_features])
            
        Returns:
            Tuple of (is_attack, confidence)
        """
        if not self.binary_session:
            raise ValueError("Binary model not loaded")
        
        # Run inference
        input_name = self.binary_session.get_inputs()[0].name
        output = self.binary_session.run(None, {input_name: features})[0]
        
        # Get prediction and confidence
        probabilities = self._softmax(output[0])
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        
        # In training: 0=Normal, 1=Attack
        is_attack = bool(prediction == 1)
        
        return is_attack, confidence
    
    def predict_multiclass(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Make multiclass classification prediction (attack type).
        
        Args:
            features: Preprocessed feature array (shape: [1, n_features])
            
        Returns:
            Tuple of (attack_type, confidence, all_probabilities)
        """
        if not self.multiclass_session:
            raise ValueError("Multiclass model not loaded")
        
        # Run inference
        input_name = self.multiclass_session.get_inputs()[0].name
        output = self.multiclass_session.run(None, {input_name: features})[0]
        
        # Get prediction and confidence
        probabilities = self._softmax(output[0])
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        
        # Map prediction index to attack type name
        # 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
        attack_type = self.attack_types[prediction] if prediction < len(self.attack_types) else "Unknown"
        
        # Create probability dictionary
        all_probabilities = {
            attack_type: float(prob)
            for attack_type, prob in zip(self.attack_types, probabilities)
        }
        
        return attack_type, confidence, all_probabilities
    
    def predict(self, features: Dict) -> Dict:
        """
        Make both binary and multiclass predictions.
        
        Args:
            features: Dictionary of network traffic features
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess features
        processed_features = self.preprocess_features(features)
        
        # Binary prediction
        is_attack, binary_confidence = self.predict_binary(processed_features)
        
        # Multiclass prediction
        attack_type, multiclass_confidence, all_probabilities = self.predict_multiclass(processed_features)
        
        # Inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        return {
            "is_attack": is_attack,
            "binary_confidence": binary_confidence,
            "attack_type": attack_type,
            "multiclass_confidence": multiclass_confidence,
            "multiclass_probabilities": all_probabilities,
            "inference_time_ms": inference_time_ms
        }
    
    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """
        Make predictions for a batch of samples.
        Optimized to run preprocessing once for the whole batch.
        """
        start_time = time.time()
        
        if not features_list:
            return []
            
        # Preprocess entire batch at once
        processed_batch = self.preprocess_batch(features_list)
        
        results = []
        
        # Run inference for each sample
        # Note: We could also batch the inference if models support it (usually they do),
        # but we'll keep it simple for now or optimize if needed.
        # ONNX runtime supports batch inference.
        
        # Binary Batch Inference
        if self.binary_session:
            input_name = self.binary_session.get_inputs()[0].name
            binary_outputs = self.binary_session.run(None, {input_name: processed_batch})[0]
        
        # Multiclass Batch Inference
        if self.multiclass_session:
            input_name = self.multiclass_session.get_inputs()[0].name
            multiclass_outputs = self.multiclass_session.run(None, {input_name: processed_batch})[0]
            
        for i in range(len(features_list)):
            sample_start = time.time()
            
            # Process binary result
            b_probs = self._softmax(binary_outputs[i])
            b_pred = int(np.argmax(b_probs))
            is_attack = bool(b_pred == 1)
            binary_conf = float(b_probs[b_pred])
            
            # Process multiclass result
            m_probs = self._softmax(multiclass_outputs[i])
            m_pred = int(np.argmax(m_probs))
            attack_type = self.attack_types[m_pred] if m_pred < len(self.attack_types) else "Unknown"
            multiclass_conf = float(m_probs[m_pred])
            all_probs = {
                at: float(prob)
                for at, prob in zip(self.attack_types, m_probs)
            }
            
            results.append({
                "is_attack": is_attack,
                "binary_confidence": binary_conf,
                "attack_type": attack_type,
                "multiclass_confidence": multiclass_conf,
                "multiclass_probabilities": all_probs,
                "inference_time_ms": (time.time() - sample_start) * 1000 # Per-sample time approx
            })
            
        return results
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax to get probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


@lru_cache
def get_ml_service() -> MLInferenceService:
    """Get cached ML inference service instance."""
    return MLInferenceService()
