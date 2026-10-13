"""
Neural network models and training utilities for multi-class classification.

Extracted and generalized from GAAL_Tunnel_427J notebook.
Includes N-class CNN classifier, device-aware data loader, and early stopping.

Key change from 427J → 511J:
    - CNNClassifier output: nn.Linear(256, 2) → nn.Linear(256, num_classes)
    - Everything else (architecture, EarlyStopping, DeviceDataLoader) is identical.

Design Notes:
-------------
The CNNClassifier is a standalone CNN trained independently from BAGAN.
There is NO weight transfer between BAGAN and CNNClassifier.

Weight transfer in this project happens ONLY between VAE and BAGAN:
    1. VAE encoder → BAGAN discriminator (feature extraction layers D_e)
    2. VAE decoder → BAGAN generator

The discriminator structure is D = (D_e, D_c) where:
    - D_e: convolutional feature extraction layers (initialized from VAE encoder)
    - D_c: classification head outputting n+1 classes (randomly initialized)

That's it. No other weight transfers occur (NOT to classifier, NOT from discriminator).

CNNClassifier uses modern CNN practices (strided convolutions, BatchNorm, LeakyReLU)
which are standard for image classification tasks, NOT because of BAGAN.

The input size is 64×64 (not 28×28) because Fashion-MNIST images are resized
to match BAGAN's output resolution for fair comparison in ablation studies.
"""

from typing import Iterator, Any, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

print("models.py loaded")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Move tensors to device
# ─────────────────────────────────────────────────────────────────────────────

def to_device(data, device):
    """Move tensor(s) to the specified device.
    
    Handles single tensors, tuples, and lists recursively.
    Used by DeviceDataLoader to move batches to GPU/CPU.
    
    Args:
        data: A tensor, tuple of tensors, or list of tensors
        device: Target device string ('cuda' or 'cpu')
    
    Returns:
        Same structure with all tensors moved to device
    
    How it works:
        - If data is a list/tuple (like a batch of (images, labels)),
          recursively move each element to the device.
        - If data is a single tensor, call .to(device) on it.
    
    Example:
        >>> images, labels = to_device((images, labels), 'cuda')
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    #                      ^^^^^^^^^^^^^^^^
    # non_blocking=True: Don't wait for the transfer to complete before
    # continuing CPU execution. This allows CPU and GPU to work in parallel,
    # improving throughput when using pin_memory=True in DataLoader.


# ─────────────────────────────────────────────────────────────────────────────
# CNNClassifier: N-class convolutional neural network
# ─────────────────────────────────────────────────────────────────────────────

class CNNClassifier(nn.Module):
    """Convolutional Neural Network for N-class Fashion-MNIST classification.
    
    This is a standalone classifier trained independently from BAGAN.
    NO weight transfer happens between BAGAN and this classifier.
    
    Architecture (4 conv blocks — no BatchNorm for 514J):
    
        Input: (batch, 1, 64, 64)          ← Fashion-MNIST resized to 64×64
                    │
        ┌───────────┴───────────┐
        │   Conv Block 1        │  Conv2d(1→16, 3×3, stride=2) + LeakyReLU
        │   Output: (batch, 16, 32, 32)
        ├───────────────────────┤
        │   Conv Block 2        │  Conv2d(16→32, 3×3, stride=2) + LeakyReLU
        │   Output: (batch, 32, 16, 16)
        ├───────────────────────┤
        │   Conv Block 3        │  Conv2d(32→64, 3×3, stride=2) + LeakyReLU
        │   Output: (batch, 64, 8, 8)
        ├───────────────────────┤
        │   Conv Block 4        │  Conv2d(64→128, 3×3, stride=2) + LeakyReLU
        │   Output: (batch, 128, 4, 4)
        ├───────────────────────┤
        │   Flatten             │  (batch, 128×4×4) = (batch, 2048)
        │   FC1: 2048 → 64      │  Linear + LeakyReLU + Dropout
        │   FC2: 64 → N         │  Linear (N = num_classes)
        └───────────────────────┘
                    │
        Output: (batch, num_classes)       ← Raw logits (not probabilities!)
    
    Architecture rationale (modern CNN best practices):
    ---------------------------------------------------
    Why stride=2 instead of MaxPool?
        Strided convolutions let the network LEARN how to downsample,
        rather than using fixed max-pooling. Standard in modern CNNs
        (DCGAN, ResNet variants) for better gradient flow and fewer
        artifacts.
    
    Why LeakyReLU(0.2) instead of ReLU?
        LeakyReLU allows small negative gradients (slope=0.2) to flow,
        preventing "dying ReLU" where neurons output zero forever.
        More robust for deep networks, especially with batch norm.
    
    Why NO BatchNorm? (514J change)
        BatchNorm normalizes activations per-channel across the batch,
        which stabilizes training and acts as mild regularization.
        Removing it makes the classifier MORE SENSITIVE to class imbalance:
        - Without BN, majority-class gradients dominate updates
        - Minority-class features get less stable gradient signal
        - Ab-1 (imbalanced) suffers more; Ab-4 (balanced via BAGAN) is
          less affected because balanced batches provide balanced gradients
        - Net effect: widens the performance gap between Ab-1 and Ab-4
        Previously (513J) with BN, the classifier was too robust—it
        learned minority patterns even with only 5% data.
    
    Why bias=True in Conv layers? (514J change)
        In 513J, bias=False was used because BatchNorm's learnable beta
        parameter made conv bias redundant. Without BatchNorm, the conv
        layers need their own bias term for proper function approximation.
    
    Args:
        input_channels: Number of input channels (1 for grayscale Fashion-MNIST)
        num_classes: Number of output classes.
                 Current default is 3 for the T-shirt/top, Pullover, Dress setup.
        dropout_rate: Dropout probability for regularization (default: 0.3)
    
    Example:
        >>> model = CNNClassifier(num_classes=3)
        >>> x = torch.randn(32, 1, 64, 64)  # batch of 32 grayscale 64x64 images
        >>> logits = model(x)
        >>> print(logits.shape)
        torch.Size([32, 3])
        >>> # To get probabilities: probs = F.softmax(logits, dim=1)
        >>> # To get predictions:   preds = logits.argmax(dim=1)
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 3, dropout_rate: float = 0.3):
        """Initialize CNN classifier layers.
        
        The network is built in two parts:
        1. Feature extractor: 4 conv blocks that progressively extract features
           (1×64×64) → (16×32×32) → (32×16×16) → (64×8×8) → (128×4×4)
        2. Classifier head: FC layers that map features to class logits
           2048 → 64 → num_classes (bottleneck forces selective features)
        """
        super(CNNClassifier, self).__init__()
        
        # Store config for reference (useful for debugging / model inspection)
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # ── Feature Extractor (4 convolutional blocks, no BatchNorm) ──────
        # 514J: Removed BatchNorm to make classifier more sensitive to
        # class imbalance. Re-enabled bias=True since BN beta is gone.
        
        # Block 1: Input (batch, 1, 64, 64) → Output (batch, 16, 32, 32)
        # kernel_size=3 captures local patterns (edges, textures)
        # stride=2 halves spatial dimensions: 64/2 = 32
        # padding=1 preserves spatial size before stride reduction
        # 8C: Halved channels (was 32) to reduce capacity
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1, bias=True)
        
        # Block 2: (batch, 16, 32, 32) → (batch, 32, 16, 16)
        # Doubles channel count while halving spatial dims: 32/2 = 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True)
        
        # Block 3: (batch, 32, 16, 16) → (batch, 64, 8, 8)
        # Continues pattern: more channels, smaller spatial dims: 16/2 = 8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True)
        
        # Block 4: (batch, 64, 8, 8) → (batch, 128, 4, 4)
        # Final conv block: 8/2 = 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        
        # Shared activation for all blocks
        # LeakyReLU(0.2): f(x) = x if x > 0, else 0.2*x
        # The 0.2 slope prevents dead neurons by allowing small negative gradients
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        #                                    ^^^^^^^^^^^^
        # inplace=True: Modifies tensor directly, saving memory
        # (no need to store both input and output)
        
        # ── Classifier Head (fully connected layers) ───────────────────
        
        # After conv blocks: 128 channels × 4 height × 4 width = 2048 features
        # 8C: Was 256*4*4=4096 — halved channels reduce feature count
        self.feature_size = 128 * 4 * 4  # = 2048
        
        # FC1: Reduce 2048 features to compact 64-dim bottleneck
        # 512J change: 256 → 64 to create information bottleneck — forces model
        # to be more selective about features, increasing reliance on data quality
        self.fc1 = nn.Linear(self.feature_size, 64)
        
        # Dropout: Randomly zero 30% of neurons during training
        # This forces the network to learn redundant representations,
        # preventing over-reliance on any single neuron → reduces overfitting
        self.dropout = nn.Dropout(dropout_rate)
        
        # FC2: Map 64 features to num_classes output logits
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image → features → class logits.
        
        Args:
            x: Input tensor [batch_size, 1, 64, 64]
               Grayscale Fashion-MNIST images resized to 64×64
        
        Returns:
            logits: Raw (unnormalized) class scores [batch_size, num_classes]
                    Use F.softmax(logits, dim=1) for probabilities
                    Use logits.argmax(dim=1) for predicted class
        
        Data flow:
            (B,1,64,64) → conv blocks → (B,128,4,4) → flatten → (B,2048)
            → FC1 → (B,64) → dropout → FC2 → (B, num_classes)
        """
        # ── Conv Block 1: Extract low-level features (edges, textures) ──
        x = self.conv1(x)       # (B, 1, 64, 64) → (B, 16, 32, 32)
        x = self.activation(x)  # Apply LeakyReLU non-linearity
        
        # ── Conv Block 2: Combine low-level features into patterns ──────
        x = self.conv2(x)       # (B, 16, 32, 32) → (B, 32, 16, 16)
        x = self.activation(x)
        
        # ── Conv Block 3: Detect higher-level shapes and structures ─────
        x = self.conv3(x)       # (B, 32, 16, 16) → (B, 64, 8, 8)
        x = self.activation(x)
        
        # ── Conv Block 4: Capture abstract, class-discriminative features ─
        x = self.conv4(x)       # (B, 64, 8, 8) → (B, 128, 4, 4)
        x = self.activation(x)
        
        # ── Flatten: Reshape 3D feature maps → 1D feature vector ────────
        x = x.view(x.size(0), -1)  # (B, 128, 4, 4) → (B, 2048)
        # x.size(0) = batch size, -1 = auto-compute remaining (128*4*4 = 2048)
        
        # ── Classification Head ─────────────────────────────────────────
        x = self.fc1(x)         # (B, 2048) → (B, 64)  ← bottleneck
        x = self.activation(x)  # Non-linearity
        x = self.dropout(x)     # Regularization (only active during model.train())
        x = self.fc2(x)         # (B, 64) → (B, num_classes)
        
        return x  # Raw logits — CrossEntropyLoss expects raw logits, not softmax

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate features: image → 64-dim post-fc1 activation (pre-fc2).
        
        Used by Ab-11a feature matching loss (φ in `λ_fm · ‖φ_real.mean −
        φ_fake.mean‖²`). Mirrors `forward()` exactly through fc1 +
        activation, but stops *before* dropout and fc2 so that:
            (a) the output is deterministic w.r.t. dropout (caller may still
                want to run under model.eval() to disable BN running stats),
            (b) the returned vector lies in the same 64-dim representation
                space the classifier head consumes, which is the most
                semantically meaningful "what the classifier sees" signal.
        
        Args:
            x: Input tensor [batch_size, 1, 64, 64].
        
        Returns:
            features: [batch_size, 64] post-activation pre-fc2 features.
        """
        x = self.conv1(x); x = self.activation(x)
        x = self.conv2(x); x = self.activation(x)
        x = self.conv3(x); x = self.activation(x)
        x = self.conv4(x); x = self.activation(x)
        x = x.view(x.size(0), -1)   # (B, 2048)
        x = self.fc1(x)             # (B, 64)
        x = self.activation(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# DeviceDataLoader: Automatically move batches to GPU/CPU
# ─────────────────────────────────────────────────────────────────────────────

class DeviceDataLoader:
    """Wrapper that moves DataLoader batches to a device (GPU/CPU) automatically.
    
    Without this, you'd have to write `images = images.to(device)` and
    `labels = labels.to(device)` at the start of every training loop iteration.
    DeviceDataLoader handles this transparently.
    
    How it works:
        1. Wraps an existing PyTorch DataLoader
        2. When you iterate over it, each batch is moved to the device
           BEFORE yielding, so the training loop receives GPU-ready tensors
    
    Args:
        dataloader: Standard PyTorch DataLoader
        device: Target device string ('cuda' or 'cpu')
    
    Example:
        Without DeviceDataLoader (manual transfer):
        >>> for images, labels in dataloader:
        ...     images = images.to('cuda')   # Manual!
        ...     labels = labels.to('cuda')   # Manual!
        ...     output = model(images)
        
        With DeviceDataLoader (automatic transfer):
        >>> dl = DeviceDataLoader(dataloader, 'cuda')
        >>> for images, labels in dl:
        ...     output = model(images)  # Already on GPU!
    """
    
    def __init__(self, dataloader: DataLoader, device: str):
        """Store the dataloader and target device.
        
        Args:
            dataloader: PyTorch DataLoader to wrap
            device: Device to move batches to ('cuda' or 'cpu')
        """
        self.dl = dataloader    # The underlying DataLoader
        self.device = device    # Where to send the data
    
    def __iter__(self) -> Iterator[Any]:
        """Yield batches with all tensors moved to self.device.
        
        Each batch from the DataLoader is typically a tuple of (images, labels).
        The to_device() helper recursively moves each tensor in the tuple to
        the target device.
        
        Yields:
            Batch with all tensors on self.device
        """
        for batch in self.dl:
            yield to_device(batch, self.device)
    
    def __len__(self) -> int:
        """Return number of batches (same as underlying DataLoader)."""
        return len(self.dl)


# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping: Stop training when validation loss plateaus
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Early stopping to prevent overfitting during classifier training.
    
    How it works:
        1. After each epoch, you pass the current validation loss and model
        2. If val_loss improved (decreased by at least min_delta):
           - Save a copy of the model weights (best_model)
           - Reset the patience counter
        3. If val_loss did NOT improve:
           - Increment the patience counter
        4. If counter reaches patience limit:
           - Restore the best model weights
           - Return True to signal "stop training"
    
    Why restore best weights?
        Without this, the final model would be from the LAST epoch, which
        often has higher validation loss than the BEST epoch. Restoring
        ensures we keep the model that generalized best.
    
    Visual timeline example (patience=3):
        Epoch:    1    2    3    4    5    6    7
        Val loss: 0.5  0.4  0.35 0.36 0.37 0.38 STOP!
        Counter:  0    0    0    1    2    3    → triggered
        Best:     0.5  0.4  0.35 0.35 0.35 0.35 ← restored to epoch 3
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
                  (default: 5). Higher = more patient, might train longer.
        min_delta: Minimum change to count as improvement
                   (default: 0). Set > 0 to ignore tiny improvements.
        restore_best_weights: Whether to restore best model when stopping
                              (default: True). Almost always want this True.
          mode: 'min' to minimize the monitored metric (e.g., val_loss),
              'max' to maximize it (e.g., macro_f2).
    
    Example:
        >>> es = EarlyStopping(patience=7, min_delta=0.001, mode='max')
        >>> for epoch in range(100):
        ...     val_loss, metrics = validate(model, val_loader)
        ...     if es(model, metrics['macro_f2']):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break  # model already has best weights restored
    """
    
    def __init__(self, patience: int = 5, min_delta: float = 0, restore_best_weights: bool = True, mode: str = 'min'):
        """Initialize early stopping tracker.
        
        Args:
            patience: How many epochs without improvement before stopping
            min_delta: Minimum change to count as improvement
            restore_best_weights: Whether to restore best model on stop
            mode: 'min' to minimize (e.g., val_loss) or 'max' to maximize (e.g., macro_f2)
        """
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        # State variables (updated each epoch)
        self.best_model = None   # Deep copy of best model's state_dict
        self.best_loss = None    # Best monitored metric value seen so far
        self.counter = 0         # Epochs since last improvement
        self.status = ""         # Human-readable status message
        self.improved = False    # True if last __call__ found improvement
        self.best_epoch = 0      # Epoch number of best metric (for avg stopping epoch)
    
    def __call__(self, model: nn.Module, val_metric: float, epoch: int = 0) -> bool:
        """Check if training should stop.
        
        Called once per epoch with the current model and monitored metric.
        
        Args:
            model: Current model (needed to save/restore weights)
            val_metric: Current epoch's monitored metric value
                        (val_loss if mode='min', macro_f2 if mode='max')
            epoch: Current epoch number (0-indexed). Tracked so that
                   after training, self.best_epoch tells us which epoch
                   had the best metric — used to set retrain duration.
        
        Returns:
            True if training should stop (patience exhausted)
            False if training should continue
        
        Side effect:
            If returning True and restore_best_weights=True,
            model's weights are restored to the best epoch's weights.
            self.improved is set to True when a new best is found.
        """
        self.improved = False
        
        if self.best_loss is None:
            # ── First epoch: Initialize baseline ────────────────────────
            self.best_loss = val_metric
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch  # Track which epoch was best
            self.status = f"Initial epoch"
            self.improved = True
            
        elif self._is_improvement(val_metric):
            # ── Improvement found! ──────────────────────────────────────
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_metric
            self.best_epoch = epoch  # Update: this epoch is the new best
            self.counter = 0  # Reset patience counter
            self.status = f"Improvement found, counter reset to {self.counter}"
            self.improved = True
            
        else:
            # ── No improvement ──────────────────────────────────────────
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            
            if self.counter >= self.patience:
                # ── Patience exhausted → STOP ───────────────────────────
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    # Restore model to best epoch's weights
                    model.load_state_dict(self.best_model)
                return True  # Signal: stop training
        
        return False  # Signal: continue training
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current metric is an improvement over the best.
        
        For mode='min': improvement means current < best - min_delta
        For mode='max': improvement means current > best + min_delta
        """
        if self.mode == 'min':
            return self.best_loss - current >= self.min_delta
        return current - self.best_loss >= self.min_delta
    
    def reset(self):
        """Reset early stopping state for a new training run.
        
        Call this when starting a new fold in k-fold CV to ensure
        clean state (no carryover from previous fold).
        """
        self.best_model = None
        self.best_loss = None
        self.best_epoch = 0     # Reset best epoch for new training run
        self.counter = 0
        self.status = ""
        self.improved = False
