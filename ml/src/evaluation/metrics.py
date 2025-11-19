"""
Model evaluation metrics and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_recall_fscore_support
)
from typing import Dict, List, Tuple
import pandas as pd


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    title: str = 'ROC Curves',
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels (one-hot encoded for multiclass)
        y_proba: Predicted probabilities
        classes: Class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_classes = len(classes)
    
    # For binary classification
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', linewidth=2)
    else:
        # For multiclass
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.4f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    title: str = 'Precision-Recall Curves',
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plot precision-recall curves for each class.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        classes: Class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_classes = len(classes)
    
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        ax.plot(recall, precision, label=f'{classes[1]}', linewidth=2)
    else:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        for i, class_name in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            ax.plot(recall, precision, label=class_name, linewidth=2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str]
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        
    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'per_class': {},
        'confusion_matrix': cm.tolist(),
        'support': support.tolist() if hasattr(support, 'tolist') else support
    }
    
    for i, class_name in enumerate(classes):
        metrics['per_class'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }
    
    return metrics


def print_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    title: str = "Classification Report"
):
    """
    Print comprehensive classification summary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        title: Report title
    """
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)
    
    print("\nClassification Report:")
    print("-" * 80)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    
    print("\nConfusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(y_true, y_pred)
    
    # Create DataFrame for better display
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Evaluation metrics module loaded successfully!")







