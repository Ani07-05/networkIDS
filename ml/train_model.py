"""
Standalone script to train NIDS models on NSL-KDD dataset.

This script trains both binary and multiclass classifiers, evaluates them,
and exports to ONNX format for production deployment.

Usage:
    python ml/train_model.py --epochs 50 --batch-size 512 --device cuda

Requirements:
    - Full NSL-KDD dataset (run: python ml/src/data/download_dataset.py)
    - PyTorch with CUDA support (for GPU training)
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.data.preprocessing import load_data, NSLKDDPreprocessor
from src.models.hybrid_model import create_model, count_parameters
from src.training.trainer import NIDSTrainer, create_dataloaders, calculate_class_weights
from src.training.export_onnx import export_to_onnx, save_model_metadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train NIDS Models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for training (default: 512)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on (default: cuda)')
    parser.add_argument('--hidden-dims', type=int, nargs='+',
                       default=[256, 128, 64],
                       help='Hidden layer dimensions (default: 256 128 64)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--data-dir', type=str, default='ml/data',
                       help='Data directory (default: ml/data)')
    parser.add_argument('--output-dir', type=str, default='ml/models',
                       help='Output directory for models (default: ml/models)')
    return parser.parse_args()


def load_and_preprocess_data(data_dir: str):
    """Load and preprocess NSL-KDD dataset."""
    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    data_path = Path(data_dir)
    train_path = data_path / "raw" / "KDDtrain.txt"
    test_path = data_path / "raw" / "KDDtest.txt"
    
    # Check if full dataset exists
    if not train_path.exists() or not test_path.exists():
        print("[WARNING] Full dataset not found!")
        print(f"Expected files:")
        print(f"  - {train_path}")
        print(f"  - {test_path}")
        print("\nPlease download the dataset first:")
        print("  python ml/src/data/download_dataset.py")
        sys.exit(1)
    
    print(f"Loading training data from: {train_path}")
    train_df = load_data(str(train_path))
    print(f"[OK] Loaded {len(train_df):,} training samples")
    
    print(f"Loading test data from: {test_path}")
    test_df = load_data(str(test_path))
    print(f"[OK] Loaded {len(test_df):,} test samples")
    
    # Preprocess
    print("\nPreprocessing data...")
    preprocessor = NSLKDDPreprocessor()
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)
    
    # Save preprocessor
    processed_dir = data_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = processed_dir / "preprocessor.json"
    preprocessor.save(str(preprocessor_path))
    print(f"[OK] Preprocessor saved to: {preprocessor_path}")
    
    # Extract features and labels
    feature_cols = preprocessor.feature_names
    X_train = train_processed[feature_cols].values
    y_train_binary = train_processed['label_binary'].values
    y_train_multiclass = train_processed['label_multiclass'].values
    
    X_test = test_processed[feature_cols].values
    y_test_binary = test_processed['label_binary'].values
    y_test_multiclass = test_processed['label_multiclass'].values
    
    # Create train/val split
    print(f"\nCreating train/validation split (80/20)...")
    X_train_split, X_val, y_train_b, y_val_b, y_train_m, y_val_m = train_test_split(
        X_train, y_train_binary, y_train_multiclass,
        test_size=0.2,
        random_state=42,
        stratify=y_train_multiclass
    )
    
    print(f"[OK] Training set: {X_train_split.shape[0]:,} samples")
    print(f"[OK] Validation set: {X_val.shape[0]:,} samples")
    print(f"[OK] Test set: {X_test.shape[0]:,} samples")
    print(f"[OK] Features: {X_train.shape[1]}")
    
    # Print class distribution
    print("\nClass Distribution (Training Set):")
    unique_b, counts_b = np.unique(y_train_b, return_counts=True)
    for cls, cnt in zip(unique_b, counts_b):
        print(f"  Binary Class {cls}: {cnt:,} ({cnt/len(y_train_b)*100:.1f}%)")
    
    unique_m, counts_m = np.unique(y_train_m, return_counts=True)
    class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    for cls, cnt in zip(unique_m, counts_m):
        print(f"  {class_names[cls]}: {cnt:,} ({cnt/len(y_train_m)*100:.1f}%)")
    
    return {
        'X_train': X_train_split, 'X_val': X_val, 'X_test': X_test,
        'y_train_b': y_train_b, 'y_val_b': y_val_b, 'y_test_b': y_test_binary,
        'y_train_m': y_train_m, 'y_val_m': y_val_m, 'y_test_m': y_test_multiclass,
        'preprocessor': preprocessor
    }


def train_binary_model(data, args, output_dir):
    """Train binary classification model."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING BINARY CLASSIFIER (Normal vs Attack)")
    print("=" * 70)
    
    # Calculate class weights
    class_weights = calculate_class_weights(data['y_train_b'])
    print(f"Class weights: {class_weights.numpy()}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data['X_train'], data['y_train_b'],
        data['X_val'], data['y_val_b'],
        batch_size=args.batch_size
    )
    
    # Create model
    input_dim = data['X_train'].shape[1]
    model = create_model('binary', input_dim, args.hidden_dims, args.dropout)
    print(f"Model created with {count_parameters(model):,} parameters")
    print(f"Architecture: Input({input_dim}) -> {' -> '.join(map(str, args.hidden_dims))} -> Output(2)")
    
    # Train
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = NIDSTrainer(model, device=device, learning_rate=args.learning_rate)
    history = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        task='binary',
        class_weights=class_weights,
        early_stopping_patience=args.patience,
        checkpoint_dir=output_dir
    )
    
    # Load best model and evaluate on test
    trainer.load_checkpoint(str(Path(output_dir) / "best_model_binary.pth"))
    model.eval()
    model.to('cpu')
    
    X_test_tensor = torch.FloatTensor(data['X_test'])
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).numpy()
    
    # Calculate metrics
    acc = accuracy_score(data['y_test_b'], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        data['y_test_b'], preds, average='weighted'
    )
    cm = confusion_matrix(data['y_test_b'], preds)
    
    print("\n" + "-" * 70)
    print("BINARY MODEL - TEST SET RESULTS:")
    print("-" * 70)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(data['y_test_b'], preds,
                                target_names=['Normal', 'Attack'], digits=4))
    
    return {
        'model': model,
        'history': history,
        'test_metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
        'confusion_matrix': cm,
        'predictions': preds
    }


def train_multiclass_model(data, args, output_dir):
    """Train multiclass classification model."""
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING MULTICLASS CLASSIFIER (5 Attack Types)")
    print("=" * 70)
    
    # Calculate class weights
    class_weights = calculate_class_weights(data['y_train_m'])
    print(f"Class weights: {class_weights.numpy()}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data['X_train'], data['y_train_m'],
        data['X_val'], data['y_val_m'],
        batch_size=args.batch_size
    )
    
    # Create model
    input_dim = data['X_train'].shape[1]
    model = create_model('multiclass', input_dim, args.hidden_dims, args.dropout)
    print(f"Model created with {count_parameters(model):,} parameters")
    print(f"Architecture: Input({input_dim}) -> {' -> '.join(map(str, args.hidden_dims))} -> Output(5)")
    
    # Train
    device = args.device if torch.cuda.is_available() else 'cpu'
    trainer = NIDSTrainer(model, device=device, learning_rate=args.learning_rate)
    history = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        task='multiclass',
        class_weights=class_weights,
        early_stopping_patience=args.patience,
        checkpoint_dir=output_dir
    )
    
    # Load best model and evaluate on test
    trainer.load_checkpoint(str(Path(output_dir) / "best_model_multiclass.pth"))
    model.eval()
    model.to('cpu')
    
    X_test_tensor = torch.FloatTensor(data['X_test'])
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = torch.argmax(outputs, dim=1).numpy()
    
    # Calculate metrics
    acc = accuracy_score(data['y_test_m'], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        data['y_test_m'], preds, average='weighted'
    )
    cm = confusion_matrix(data['y_test_m'], preds)
    
    # Per-class metrics
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        data['y_test_m'], preds, average=None
    )
    
    print("\n" + "-" * 70)
    print("MULTICLASS MODEL - TEST SET RESULTS:")
    print("-" * 70)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(data['y_test_m'], preds,
                                target_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
                                digits=4))
    
    per_class_metrics = {
        'normal': {'precision': float(per_class_prec[0]), 'recall': float(per_class_rec[0]), 'f1': float(per_class_f1[0])},
        'dos': {'precision': float(per_class_prec[1]), 'recall': float(per_class_rec[1]), 'f1': float(per_class_f1[1])},
        'probe': {'precision': float(per_class_prec[2]), 'recall': float(per_class_rec[2]), 'f1': float(per_class_f1[2])},
        'r2l': {'precision': float(per_class_prec[3]), 'recall': float(per_class_rec[3]), 'f1': float(per_class_f1[3])},
        'u2r': {'precision': float(per_class_prec[4]), 'recall': float(per_class_rec[4]), 'f1': float(per_class_f1[4])}
    }
    
    return {
        'model': model,
        'history': history,
        'test_metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1},
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm,
        'predictions': preds
    }


def export_models(binary_result, multi_result, data, args):
    """Export models to ONNX format."""
    print("\n" + "=" * 70)
    print("STEP 4: EXPORTING MODELS TO ONNX")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dim = data['X_train'].shape[1]
    
    # Export binary model
    print("\nExporting binary model...")
    export_to_onnx(
        binary_result['model'],
        str(output_dir / "nids_model_binary.onnx"),
        input_dim,
        "Binary NIDS Model"
    )
    
    # Export multiclass model
    print("\nExporting multiclass model...")
    export_to_onnx(
        multi_result['model'],
        str(output_dir / "nids_model_multiclass.onnx"),
        input_dim,
        "Multiclass NIDS Model"
    )
    
    # Save metadata
    print("\nSaving model metadata...")
    model_metadata = {
        "input_dim": input_dim,
        "hidden_dims": args.hidden_dims,
        "dropout_rate": args.dropout,
        "model_type": "deep_neural_network",
        "framework": "pytorch",
        "binary_params": count_parameters(binary_result['model']),
        "multiclass_params": count_parameters(multi_result['model']),
        "training_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    }
    
    metrics = {
        "binary": {
            **{k: float(v) for k, v in binary_result['test_metrics'].items()},
            "confusion_matrix": binary_result['confusion_matrix'].tolist()
        },
        "multiclass": {
            **{k: float(v) for k, v in multi_result['test_metrics'].items()},
            "per_class": multi_result['per_class_metrics'],
            "confusion_matrix": multi_result['confusion_matrix'].tolist()
        }
    }
    
    preprocessing_info = {
        "num_features": len(data['preprocessor'].feature_names),
        "scaler": "StandardScaler",
        "categorical_encoding": "one-hot",
        "categorical_features": ["protocol_type", "service", "flag"],
        "preprocessor_path": str(Path(args.data_dir) / "processed" / "preprocessor.json")
    }
    
    save_model_metadata(
        str(output_dir / "model_metadata.json"),
        model_metadata,
        metrics,
        preprocessing_info
    )
    
    print(f"\n[OK] All models and metadata saved to: {output_dir}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("NETWORK INTRUSION DETECTION SYSTEM - MODEL TRAINING")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Hidden Layers: {args.hidden_dims}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Device: {args.device}")
    print(f"  Early Stopping Patience: {args.patience}")
    
    # Check CUDA
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("  [WARNING] CUDA not available, falling back to CPU")
            args.device = 'cpu'
    
    # Load and preprocess data
    data = load_and_preprocess_data(args.data_dir)
    
    # Train models
    binary_result = train_binary_model(data, args, args.output_dir)
    multi_result = train_multiclass_model(data, args, args.output_dir)
    
    # Export models
    export_models(binary_result, multi_result, data, args)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("=" * 70)
    print("\nBinary Classifier (Normal vs Attack):")
    print(f"  Test Accuracy:  {binary_result['test_metrics']['accuracy']:.4f}")
    print(f"  Test F1-Score:  {binary_result['test_metrics']['f1']:.4f}")
    
    print("\nMulticlass Classifier (5 Classes):")
    print(f"  Test Accuracy:  {multi_result['test_metrics']['accuracy']:.4f}")
    print(f"  Test F1-Score:  {multi_result['test_metrics']['f1']:.4f}")
    
    print("\nPer-Class Performance:")
    for cls_name, metrics in multi_result['per_class_metrics'].items():
        print(f"  {cls_name.capitalize():8s}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    print(f"\nModels saved in: {args.output_dir}/")
    print("  - nids_model_binary.onnx")
    print("  - nids_model_multiclass.onnx")
    print("  - best_model_binary.pth")
    print("  - best_model_multiclass.pth")
    print("  - model_metadata.json")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] READY FOR PRODUCTION DEPLOYMENT")
    print("=" * 70)


if __name__ == "__main__":
    main()

