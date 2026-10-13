"""
Data loading and preprocessing utilities for imbalanced multi-class datasets.

Handles Fashion-MNIST dataset loading, class filtering, and imbalanced pool creation.
Replaces filter_and_convert_labels from 427J with multi-class support.

Design Notes:
-------------
- create_imbalanced_pool() returns (dataset, indices) because the pool is split
  into k folds, and StratifiedKFold.split() requires indices for splitting.
  
- build_test_set() returns only dataset (no indices) because the test set is
  used as-is for evaluation without any splitting or fold creation.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, TensorDataset
import numpy as np
import torchvision
import torchvision.transforms as transforms

print("data.py loaded")

def filter_by_classes(
    dataset: Dataset,
    classes: List[int],
    proportions: List[float],
    seed: int = 42,
    max_samples_majority: Optional[int] = None
) -> Tuple[Subset, np.ndarray]:
    """Filter dataset to selected classes with specified proportions.
    
    Unlike 427J's filter_and_convert_labels, this preserves original labels
    and uses seeded randomness for reproducibility.
    
    Args:
        dataset: PyTorch dataset to filter
        classes: List of class labels to keep
        proportions: Proportion for each class (first class is majority, proportion=1.0)
        seed: Random seed for reproducible sampling
        max_samples_majority: If set, cap majority class at this count.
            Minority counts are computed relative to the capped count.
            E.g., max_samples_majority=800, minority prop=0.05 → 40 samples.
    
    Returns:
        filtered_dataset: Subset of original dataset
        filtered_indices: Original indices of selected samples
    
    Example:
        >>> # Default capped setup: 800 majority, 40 minority (ratio 0.05)
        >>> filtered_data, indices = filter_by_classes(
        ...     train_data, [0, 2, 3], [1.0, 0.05, 0.05], seed=42,
        ...     max_samples_majority=800
        ... )
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # Get all labels from dataset
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, 'tensors'):
        all_labels = dataset.tensors[1].numpy()
    else:
        all_labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Find indices for each class
    class_indices = {}
    for c in classes:
        class_indices[c] = np.where(all_labels == c)[0]
    
    # Find majority class sample count (class with proportion=1.0)
    majority_idx = proportions.index(max(proportions))
    majority_class = classes[majority_idx]
    n_majority = len(class_indices[majority_class])
    
    # 514J: Cap majority count if max_samples_majority is set.
    # This ensures minority = prop * capped_count (not prop * all_available).
    if max_samples_majority is not None:
        n_majority = min(n_majority, max_samples_majority)
    
    # Sample from each class according to proportions
    selected_indices = []
    for c, prop in zip(classes, proportions):
        indices = class_indices[c]
        # Calculate target count based on (capped) majority size
        if prop >= 1.0:
            # Majority class: use capped count
            n_samples = n_majority
        else:
            # Sample proportionally for minority classes
            n_samples = int(prop * n_majority)
            n_samples = min(n_samples, len(indices))  # Can't sample more than available
        
        if n_samples < len(indices):
            sampled = rng.choice(indices, size=n_samples, replace=False)
        else:
            sampled = indices
        selected_indices.extend(sampled)
    
    selected_indices = np.array(sorted(selected_indices))
    
    # Create subset with selected indices
    filtered_dataset = Subset(dataset, selected_indices.tolist())
    
    return filtered_dataset, selected_indices


def create_imbalanced_pool(config) -> Tuple[Subset, np.ndarray]:
    """Create imbalanced training pool from Fashion-MNIST train split.
    
    Returns pool_indices in addition to the dataset because:
    - The pool is used for k-fold cross-validation which requires splitting
    - Indices are needed by StratifiedKFold.split() to create train/val folds
    - Useful for tracking data provenance and debugging fold composition
    
    Args:
        config: AblationConfig instance with selected_classes and class_proportions
    
    Returns:
        pool_dataset: Imbalanced training dataset (Subset of FashionMNIST)
        pool_indices: Original indices in Fashion-MNIST train set [N]
                      Required for k-fold splitting in run_kfold_ablation()
    
    Example:
        >>> config = AblationConfig(selected_classes=[0,2,3], class_proportions=[1.0,0.05,0.05])
        >>> pool_data, pool_indices = create_imbalanced_pool(config)
        >>> print(get_class_distribution(pool_data))
        {0: 800, 2: 40, 3: 40}  # Example counts with max_samples_majority=800
        >>> # Use pool_indices for k-fold splitting:
        >>> skf = StratifiedKFold(n_splits=5)
        >>> for train_idx, val_idx in skf.split(pool_indices, labels):
        ...     # train_idx, val_idx are indices into pool_indices array
        ...     pass
    """
    # Load Fashion-MNIST training data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Filter to selected classes with specified proportions
    pool_dataset, pool_indices = filter_by_classes(
        dataset=train_dataset,
        classes=config.selected_classes,
        proportions=config.class_proportions,
        seed=config.random_seed,
        max_samples_majority=config.max_samples_majority
    )
    
    return pool_dataset, pool_indices


def build_test_set(config) -> Subset:
    """Build imbalanced test set from Fashion-MNIST test split.
    
    Uses same class proportions as training pool for fair evaluation.
    
    Does NOT return indices (unlike create_imbalanced_pool) because:
    - Test set is used as-is for final evaluation (no splitting needed)
    - Consumed as a whole dataset via DataLoader for evaluation
    - No need to track original indices for held-out test data
    - Simplifies the interface for evaluation functions
    
    Args:
        config: AblationConfig instance
    
    Returns:
        test_dataset: Imbalanced test dataset with same proportions as train
                      Ready to use with DataLoader for evaluation
    
    Example:
        >>> config = AblationConfig(selected_classes=[0,2,3], class_proportions=[1.0,0.05,0.05])
        >>> test_data = build_test_set(config)
        >>> print(get_class_distribution(test_data))
        {0: 1000, 2: 50, 3: 50}  # Example test counts
        >>> # Use directly with DataLoader:
        >>> test_loader = DataLoader(test_data, batch_size=64)
        >>> metrics = evaluate_on_test(model, test_loader, config, device)
    """
    # Load Fashion-MNIST test data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Filter to selected classes with specified proportions
    # Test set: no majority cap — use all available samples per class
    test_proportions = config.test_proportions if hasattr(config, 'test_proportions') else config.class_proportions
    filtered_test, _ = filter_by_classes(
        dataset=test_dataset,
        classes=config.selected_classes,
        proportions=test_proportions,
        seed=config.random_seed + 1000  # Different seed to avoid overlap
    )
    
    return filtered_test


def get_class_distribution(dataset: Dataset) -> Dict[int, int]:
    """Get class label distribution in dataset.
    
    Args:
        dataset: PyTorch dataset with (data, label) format
    
    Returns:
        Dictionary mapping class label to count
    
    Example:
        >>> dist = get_class_distribution(train_data)
        >>> print(dist)
        {0: 5400, 2: 270, 3: 270}
    """
    distribution = {}
    
    # Handle Subset wrapper
    if isinstance(dataset, Subset):
        for idx in dataset.indices:
            _, label = dataset.dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            distribution[label] = distribution.get(label, 0) + 1
    elif hasattr(dataset, 'targets'):
        # Handle torchvision datasets
        labels = np.array(dataset.targets)
        for label in np.unique(labels):
            distribution[int(label)] = int(np.sum(labels == label))
    elif hasattr(dataset, 'tensors'):
        # Handle TensorDataset
        labels = dataset.tensors[1].numpy()
        for label in np.unique(labels):
            distribution[int(label)] = int(np.sum(labels == label))
    else:
        # Fallback: iterate through dataset
        for i in range(len(dataset)):
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            distribution[label] = distribution.get(label, 0) + 1
    
    return dict(sorted(distribution.items()))


def split_by_class(
    data: torch.Tensor,
    labels: torch.Tensor,
    classes: Optional[List[int]] = None
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Split data and labels by class.
    
    Helper function for augmentation strategies.
    
    Args:
        data: Tensor of data samples [N, ...]
        labels: Tensor of labels [N]
        classes: List of class labels to include (if None, uses unique labels)
    
    Returns:
        Dictionary mapping class_id -> (class_data, class_labels)
    
    Example:
        >>> class_dict = split_by_class(X_train, y_train, classes=[0, 2, 3])
        >>> majority_data, majority_labels = class_dict[0]
        >>> print(majority_data.shape, majority_labels.shape)
        torch.Size([5400, 1, 28, 28]) torch.Size([5400])
    """
    result = {}
    
    if classes is None:
        classes = torch.unique(labels).tolist()
    
    for c in classes:
        mask = labels == c
        class_data = data[mask]
        class_labels = labels[mask]
        result[c] = (class_data, class_labels)
    
    return result


def dataset_to_tensors(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a PyTorch Dataset to tensors.
    
    Useful for converting Subset or other Dataset types to raw tensors
    for augmentation and batch creation.
    
    Args:
        dataset: PyTorch Dataset with (data, label) items
    
    Returns:
        data: Tensor of images [N, 1, 28, 28]
        labels: Tensor of labels [N]
    
    Example:
        >>> pool_dataset, _ = create_imbalanced_pool(config)
        >>> pool_data, pool_labels = dataset_to_tensors(pool_dataset)
        >>> print(pool_data.shape, pool_labels.shape)
        torch.Size([6600, 1, 28, 28]) torch.Size([6600])
    """
    data_list = []
    label_list = []
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        data_list.append(img)
        label_list.append(label)
    
    data = torch.stack(data_list)
    labels = torch.tensor(label_list, dtype=torch.long)
    
    return data, labels


def print_class_distribution(dataset: Dataset, title: str = "Class Distribution") -> None:
    """Print formatted class distribution with percentages.
    
    Args:
        dataset: PyTorch Dataset
        title: Title for the printout
    
    Example:
        >>> print_class_distribution(pool_dataset, "Training Pool")
        Training Pool
        ─────────────────────────────────────────
        Class 0 (T-shirt/top): 6000 (90.91%)
        Class 2 (Pullover):    300 ( 4.55%)
        Class 3 (Dress):       300 ( 4.55%)
        ─────────────────────────────────────────
        Total: 6600 samples
    """
    from .config import FASHION_MNIST_CLASSES
    
    dist = get_class_distribution(dataset)
    total = sum(dist.values())
    
    print(f"\n{title}")
    print("─" * 45)
    
    for class_id, count in sorted(dist.items()):
        class_name = FASHION_MNIST_CLASSES.get(class_id, f"Class {class_id}")
        pct = 100.0 * count / total if total > 0 else 0
        print(f"  Class {class_id} ({class_name:12s}): {count:5d} ({pct:5.2f}%)")
    
    print("─" * 45)
    print(f"  Total: {total} samples\n")


# why do we need a reserve pool?
# - The reserve pool provides additional same-class samples that are not in the imbalanced training pool
# - This allows us to test augmentation strategies that leverage same-class data without contamination from the training pool
def create_reserve_pool(
    pool_indices: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create reserve pool disjoint from the imbalanced training pool.
    
    Loads full Fashion-MNIST train set, discovers which classes were used
    in the pool from pool_indices, and returns all same-class images NOT
    in the pool — preprocessed to 64×64, [-1,1] range, with remapped labels.
    
    Args:
        pool_indices: Indices of pool samples in Fashion-MNIST train set.
                      Used to exclude pool images and discover original classes.
    
    Returns:
        reserve_data: [M, 1, 64, 64] images in [-1, 1] range
        reserve_labels: [M] labels remapped to contiguous {0, 1, ..., N-1}
    """
    # Load full Fashion-MNIST train set
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    all_labels = np.array(train_dataset.targets)
    
    # Discover original classes from pool indices
    original_classes = sorted(set(all_labels[pool_indices].tolist()))
    
    # Find all same-class indices NOT in pool
    pool_set = set(pool_indices.tolist())
    reserve_indices = []
    for cls in original_classes:
        cls_indices = np.where(all_labels == cls)[0]
        reserve_indices.extend(i for i in cls_indices if i not in pool_set)
    reserve_indices = sorted(reserve_indices)
    
    # Convert to tensors
    reserve_subset = Subset(train_dataset, reserve_indices)
    reserve_data, reserve_labels = dataset_to_tensors(reserve_subset)
    
    # Preprocess: resize 28→64, normalize [0,1]→[-1,1]
    reserve_data = F.interpolate(
        reserve_data, size=64, mode='bilinear', align_corners=False
    )
    reserve_data = reserve_data * 2 - 1
    
    # Remap labels: original {0,2,3} → contiguous {0,1,2}
    label_map = {c: i for i, c in enumerate(original_classes)}
    reserve_labels = torch.tensor(
        [label_map[l.item()] for l in reserve_labels],
        dtype=reserve_labels.dtype
    )
    
    return reserve_data, reserve_labels
