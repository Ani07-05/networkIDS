"""
Export trained PyTorch models to ONNX format for efficient inference.
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
from typing import Dict


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_dim: int,
    model_name: str = "nids_model",
    opset_version: int = 12
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save ONNX model
        input_dim: Number of input features
        model_name: Name of the model
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_dim)
    
    # Export
    print(f"Exporting {model_name} to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"[OK] Model exported to: {output_path}")
    
    # Test inference
    test_inference_time(output_path, input_dim)


def export_hybrid_model_to_onnx(
    model: torch.nn.Module,
    output_dir: str,
    input_dim: int,
    opset_version: int = 12
):
    """
    Export hybrid model (separate binary and multiclass outputs) to two ONNX files.
    
    Args:
        model: Trained hybrid model
        output_dir: Directory to save ONNX models
        input_dim: Number of input features
        opset_version: ONNX opset version
    """
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dummy_input = torch.randn(1, input_dim)
    
    # Create wrapper classes for each output
    class BinaryWrapper(torch.nn.Module):
        def __init__(self, hybrid_model):
            super().__init__()
            self.model = hybrid_model
            
        def forward(self, x):
            binary_out, _ = self.model(x)
            return binary_out
    
    class MulticlassWrapper(torch.nn.Module):
        def __init__(self, hybrid_model):
            super().__init__()
            self.model = hybrid_model
            
        def forward(self, x):
            _, multiclass_out = self.model(x)
            return multiclass_out
    
    # Export binary model
    binary_wrapper = BinaryWrapper(model)
    binary_path = output_path / "nids_model_binary.onnx"
    export_to_onnx(binary_wrapper, str(binary_path), input_dim, "Binary Model", opset_version)
    
    # Export multiclass model
    multiclass_wrapper = MulticlassWrapper(model)
    multiclass_path = output_path / "nids_model_multiclass.onnx"
    export_to_onnx(multiclass_wrapper, str(multiclass_path), input_dim, "Multiclass Model", opset_version)


def test_inference_time(model_path: str, input_dim: int, num_iterations: int = 100):
    """
    Test ONNX model inference time.
    
    Args:
        model_path: Path to ONNX model
        input_dim: Number of input features
        num_iterations: Number of iterations for timing
    """
    import time
    
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    
    # Prepare test input
    test_input = np.random.randn(1, input_dim).astype(np.float32)
    
    # Warm-up
    for _ in range(10):
        session.run(None, {'input': test_input})
    
    # Timing
    start_time = time.time()
    for _ in range(num_iterations):
        session.run(None, {'input': test_input})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
    print(f"   Average inference time: {avg_time:.2f}ms")


def save_model_metadata(
    output_path: str,
    model_info: Dict,
    metrics: Dict,
    preprocessing_info: Dict
):
    """
    Save model metadata including architecture, metrics, and preprocessing info.
    
    Args:
        output_path: Path to save metadata JSON
        model_info: Model architecture information
        metrics: Model performance metrics
        preprocessing_info: Preprocessing pipeline information
    """
    metadata = {
        "model_info": model_info,
        "metrics": metrics,
        "preprocessing": preprocessing_info,
        "version": "1.0.0"
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Metadata saved to: {output_path}")


if __name__ == "__main__":
    print("ONNX Export Utilities")
    print("=" * 60)
    print("This module provides functions to export trained PyTorch models to ONNX format.")
    print("\nTo train models, run:")
    print("  python ml/train_model.py --device cuda")
    print("\nOr use the training notebook:")
    print("  jupyter notebook ml/notebooks/03_model_training.ipynb")

