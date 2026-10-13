"""
BAGAN Visualization Utilities
=============================

Provides visualization tools for BAGAN training and analysis:
- Latent space visualization (PCA/t-SNE projections)
- Class distribution visualization in latent space
- Generated sample grids
- Real vs. fake comparison
- Training history curves
- Sample progression over epochs

N-class Support:
    The color palette auto-scales to any number of classes using matplotlib's
    tab10 colormap. Classes 0 and 1 retain their original blue/red colors
    for backward compatibility. Classes 2-9 are assigned from tab10.
    The fake class always uses gray.

Reference:
    Mariani et al. (2018) "BAGAN: Data Augmentation with Balancing GAN"
    
Date: 2026-01-07
Updated: 2026-02-15 (N-class color palette, API fixes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from ..config import BAGANConfig
    from ..training.latent_space import ClassConditionalLatentGenerator


class BAGANVisualizer:
    """
    Visualization utilities for BAGAN training and analysis.
    
    Provides methods to visualize:
    - Latent space structure and class separation
    - Generated samples in grid format
    - Comparison between real and generated images
    - Training dynamics (loss curves, accuracy)
    - Sample quality progression over epochs
    
    Attributes:
        config: BAGAN configuration (optional)
        figsize: Default figure size (width, height)
        dpi: Resolution for saved figures
        class_colors: Color palette for class visualization
        style: Matplotlib style to use
    
    Example:
        >>> visualizer = BAGANVisualizer(config)
        >>> visualizer.plot_latent_space(latent_vectors, labels)
        >>> visualizer.plot_samples_grid(generated_images, nrow=8)
    """
    
    # =========================================================================
    # N-Class Color Palette (auto-generated from matplotlib tab10 colormap)
    # =========================================================================
    # Class 0 = blue (#1f77b4) and Class 1 = red (#d62728) are kept for
    # backward compatibility with the original binary BAGAN. Classes 2-9
    # use the remaining tab10 colors. The 'fake' class always uses gray.
    #
    # tab10 order: blue, orange, green, red, purple, brown, pink, gray, olive, cyan
    # We remap:    cls0,  cls2,  cls3, cls1,  cls4,   cls5, cls6,  skip, cls7,  cls8
    # =========================================================================
    @staticmethod
    def _build_default_colors(num_classes: int = 10) -> Dict[int, str]:
        """
        Build an N-class color palette.
        
        Uses matplotlib's tab10 colormap to generate distinct colors for up
        to 10 classes. Classes 0 and 1 retain their original blue/red for
        backward compatibility.
        
        How it works:
            1. Start with tab10 — a qualitative colormap with 10 distinct colors.
            2. Convert each RGBA tuple to a hex string: (R, G, B, A) → '#RRGGBB'.
            3. Assign index 0 → tab10[0] (blue), index 3 → tab10[3] (red) to
               keep Class 0 = blue and Class 1 = red.
            4. Fill remaining class indices with the unused tab10 colors.
            5. Always add 'fake' → gray.
        
        Args:
            num_classes: Number of real classes (default: 10, max for tab10)
        
        Returns:
            Dict mapping class index (int) and 'fake' (str) to hex color strings.
            
        Example:
            >>> colors = BAGANVisualizer._build_default_colors(3)
            >>> colors[0]      # '#1f77b4' (blue)
            >>> colors[1]      # '#d62728' (red)
            >>> colors[2]      # '#ff7f0e' (orange — tab10 index 1)
            >>> colors['fake'] # '#7f7f7f' (gray)
        """
        # Get all 10 tab10 colors as hex strings
        tab10_hex = [
            '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
            for r, g, b, _ in [cm.tab10(i) for i in range(10)]
        ]
        # tab10_hex[0] = '#1f77b4' (blue)
        # tab10_hex[1] = '#ff7f0e' (orange)
        # tab10_hex[2] = '#2ca02c' (green)
        # tab10_hex[3] = '#d62728' (red)
        # tab10_hex[4] = '#9467bd' (purple)
        # tab10_hex[5] = '#8c564b' (brown)
        # tab10_hex[6] = '#e377c2' (pink)
        # tab10_hex[7] = '#7f7f7f' (gray — SKIP, reserved for fake)
        # tab10_hex[8] = '#bcbd22' (olive)
        # tab10_hex[9] = '#17becf' (cyan)
        
        colors: Dict[int, str] = {}
        
        # Class 0 → blue (tab10 index 0) — backward compatible
        colors[0] = tab10_hex[0]  # '#1f77b4'
        
        # Class 1 → red (tab10 index 3) — backward compatible
        if num_classes >= 2:
            colors[1] = tab10_hex[3]  # '#d62728'
        
        # Classes 2+ → fill from remaining tab10 colors (skip index 0, 3, 7)
        # Available: [1]=orange, [2]=green, [4]=purple, [5]=brown, [6]=pink, [8]=olive, [9]=cyan
        remaining_indices = [1, 2, 4, 5, 6, 8, 9]
        class_idx = 2  # Start assigning from class 2
        for tab_idx in remaining_indices:
            if class_idx >= num_classes:
                break
            colors[class_idx] = tab10_hex[tab_idx]
            class_idx += 1
        
        # Fake class always uses gray
        colors['fake'] = '#7f7f7f'
        
        return colors
    
    def __init__(
        self,
        config: Optional[BAGANConfig] = None,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100,
        style: str = 'seaborn-v0_8-whitegrid'
    ):
        """
        Initialize the BAGAN visualizer.
        
        Args:
            config: BAGAN configuration (optional, for default settings)
            figsize: Default figure size (width, height) in inches
            dpi: Resolution for figures
            style: Matplotlib style to use
        """
        self.config = config
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn-whitegrid')
            except:
                pass  # Use default style
        
        # Set up N-class color palette
        # _build_default_colors() returns a dict with one hex color per class
        # index plus a 'fake' entry. If config is provided, we use its
        # num_classes; otherwise default to 10 (covers up to 10 classes).
        self.num_classes = config.num_classes if config else 10
        self.class_colors = self._build_default_colors(self.num_classes)
    
    def _to_numpy(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array on CPU."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    def _normalize_images(self, images: np.ndarray) -> np.ndarray:
        """Normalize images to [0, 1] range for display."""
        # images: (N, C, H, W)
        img_min = images.min()
        img_max = images.max()
        if img_max - img_min > 1e-8:
            return (images - img_min) / (img_max - img_min)
        return images
    
    # =========================================================================
    # Latent Space Visualization
    # =========================================================================
    
    def plot_latent_space(
        self,
        latent_vectors: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        method: str = 'pca',
        show_centroids: bool = True,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize latent space with 2D projection.
        
        Projects high-dimensional latent vectors to 2D using PCA or t-SNE,
        then plots points colored by class label.
        
        Args:
            latent_vectors: Shape (N, latent_dim) - encoded representations
            labels: Shape (N,) - class labels for each point
            method: Projection method ('pca' or 'tsne')
            show_centroids: Whether to show class centroids as stars
            title: Plot title (default: "Latent Space ({method})")
            ax: Existing axes to plot on (creates new figure if None)
            figsize: Override default figure size
        
        Returns:
            Tuple of (figure, axes)
        
        Example:
            >>> z = encoder(images)
            >>> fig, ax = visualizer.plot_latent_space(z, labels)
            >>> plt.show()
        """
        # Convert to numpy
        z = self._to_numpy(latent_vectors)
        y = self._to_numpy(labels).astype(int)
        
        # Project to 2D
        if method.lower() == 'pca':
            projector = PCA(n_components=2)
            z_2d = projector.fit_transform(z)
            explained_var = projector.explained_variance_ratio_.sum() * 100
        elif method.lower() == 'tsne':
            try:
                from sklearn.manifold import TSNE
                projector = TSNE(n_components=2, random_state=42, perplexity=30)
                z_2d = projector.fit_transform(z)
                explained_var = None
            except ImportError:
                raise ImportError("t-SNE requires sklearn.manifold.TSNE")
        else:
            raise ValueError(f"Unknown projection method: {method}")
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
        else:
            fig = ax.get_figure()
        
        # Plot each class
        unique_labels = np.unique(y)
        for c in unique_labels:
            mask = (y == c)
            color = self.class_colors.get(c, f'C{c}')
            label = f'Class {c} (n={mask.sum()})'
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], 
                      c=color, label=label, alpha=0.6, s=30, edgecolors='white', linewidths=0.5)
        
        # Show class centroids
        if show_centroids:
            for c in unique_labels:
                mask = (y == c)
                centroid = z_2d[mask].mean(axis=0)
                color = self.class_colors.get(c, f'C{c}')
                ax.scatter(centroid[0], centroid[1], c=color, s=200, 
                          marker='*', edgecolors='black', linewidths=1.5,
                          label=f'μ_{c}', zorder=10)
        
        # Labels and title
        if method.lower() == 'pca' and explained_var is not None:
            ax.set_xlabel(f'PC1 ({projector.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({projector.explained_variance_ratio_[1]*100:.1f}%)')
        else:
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
        
        if title is None:
            title = f'Latent Space ({method.upper()})'
            if explained_var is not None:
                title += f' - {explained_var:.1f}% variance explained'
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_class_distributions(
        self,
        latent_generator: ClassConditionalLatentGenerator,
        n_samples: int = 500,
        method: str = 'pca',
        show_samples: bool = True,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize class-conditional latent distributions.
        
        Shows the fitted Gaussian distributions (μ_c, σ_c) for each class
        by sampling from them and projecting to 2D.
        
        Args:
            latent_generator: Fitted ClassConditionalLatentGenerator
            n_samples: Number of samples to draw from each distribution
            method: Projection method ('pca' or 'tsne')
            show_samples: Whether to show sampled points
            title: Plot title
            figsize: Figure size
        
        Returns:
            Tuple of (figure, axes)
        
        Example:
            >>> latent_gen = ClassConditionalLatentGenerator(config)
            >>> latent_gen.fit(encoder, train_loader)
            >>> visualizer.plot_class_distributions(latent_gen)
        """
        if not latent_generator.is_fitted:
            raise ValueError("ClassConditionalLatentGenerator must be fitted first")
        
        device = latent_generator.device
        all_samples = []
        all_labels = []
        
        # Sample from each class distribution
        for c in range(latent_generator.num_classes):
            samples = latent_generator.sample(class_label=c, batch_size=n_samples)
            all_samples.append(samples)
            all_labels.append(torch.full((n_samples,), c, dtype=torch.long))
        
        z = torch.cat(all_samples, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Use plot_latent_space with the sampled data
        fig, ax = self.plot_latent_space(
            z, labels, method=method, show_centroids=True,
            title=title or f'Class-Conditional Latent Distributions ({method.upper()})',
            figsize=figsize
        )
        
        return fig, ax
    
    def plot_latent_comparison(
        self,
        real_latents: Union[torch.Tensor, np.ndarray],
        real_labels: Union[torch.Tensor, np.ndarray],
        sampled_latents: Union[torch.Tensor, np.ndarray],
        sampled_labels: Union[torch.Tensor, np.ndarray],
        method: str = 'pca',
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Compare real encoded latents with sampled latents side by side.
        
        Useful for verifying that the fitted distributions match the actual
        encoded data distribution.
        
        Args:
            real_latents: Encoded real images (N, latent_dim)
            real_labels: Labels for real images (N,)
            sampled_latents: Samples from fitted distribution (M, latent_dim)
            sampled_labels: Labels for sampled points (M,)
            method: Projection method
            title: Overall figure title
            figsize: Figure size
        
        Returns:
            Tuple of (figure, list of axes)
        """
        # Convert to numpy and combine for consistent projection
        z_real = self._to_numpy(real_latents)
        y_real = self._to_numpy(real_labels).astype(int)
        z_sampled = self._to_numpy(sampled_latents)
        y_sampled = self._to_numpy(sampled_labels).astype(int)
        
        # Fit projector on combined data
        z_combined = np.vstack([z_real, z_sampled])
        if method.lower() == 'pca':
            projector = PCA(n_components=2)
        else:
            from sklearn.manifold import TSNE
            projector = TSNE(n_components=2, random_state=42)
        
        z_combined_2d = projector.fit_transform(z_combined)
        n_real = len(z_real)
        z_real_2d = z_combined_2d[:n_real]
        z_sampled_2d = z_combined_2d[n_real:]
        
        # Create side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 6))
        
        # Plot real latents
        for c in np.unique(y_real):
            mask = (y_real == c)
            color = self.class_colors.get(c, f'C{c}')
            axes[0].scatter(z_real_2d[mask, 0], z_real_2d[mask, 1],
                           c=color, label=f'Class {c}', alpha=0.6, s=30)
        axes[0].set_title('Real Encoded Latents')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot sampled latents
        for c in np.unique(y_sampled):
            mask = (y_sampled == c)
            color = self.class_colors.get(c, f'C{c}')
            axes[1].scatter(z_sampled_2d[mask, 0], z_sampled_2d[mask, 1],
                           c=color, label=f'Class {c}', alpha=0.6, s=30)
        axes[1].set_title('Sampled from Fitted Distributions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        return fig, axes
    
    # =========================================================================
    # Image Visualization
    # =========================================================================
    
    def plot_samples_grid(
        self,
        images: Union[torch.Tensor, np.ndarray],
        nrow: int = 8,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Display images in a grid format.
        
        Args:
            images: Shape (N, C, H, W) - batch of images
            nrow: Number of images per row
            title: Plot title
            figsize: Figure size
            normalize: Whether to normalize images to [0, 1]
        
        Returns:
            Tuple of (figure, axes)
        
        Example:
            >>> fake_images = generator(z)
            >>> visualizer.plot_samples_grid(fake_images, nrow=8, title='Generated Samples')
        """
        # Convert to numpy: (N, C, H, W)
        imgs = self._to_numpy(images)
        
        if normalize:
            imgs = self._normalize_images(imgs)
        
        n_images = len(imgs)
        ncol = nrow
        nrow_actual = (n_images + ncol - 1) // ncol
        
        # Determine figure size
        if figsize is None:
            # Scale based on image size and grid
            h, w = imgs.shape[2], imgs.shape[3]
            scale = min(1.5, 12 / (ncol * w / 64))
            figsize = (ncol * scale, nrow_actual * scale)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create grid
        grid_imgs = []
        for row in range(nrow_actual):
            row_imgs = []
            for col in range(ncol):
                idx = row * ncol + col
                if idx < n_images:
                    img = imgs[idx]
                    # Convert (C, H, W) to (H, W, C) or (H, W) for grayscale
                    if img.shape[0] == 1:
                        img = img[0]  # Grayscale: (H, W)
                    else:
                        img = np.transpose(img, (1, 2, 0))  # RGB: (H, W, C)
                    row_imgs.append(img)
                else:
                    # Pad with white
                    h, w = imgs.shape[2], imgs.shape[3]
                    if imgs.shape[1] == 1:
                        row_imgs.append(np.ones((h, w)))
                    else:
                        row_imgs.append(np.ones((h, w, imgs.shape[1])))
            grid_imgs.append(np.concatenate(row_imgs, axis=1))
        
        grid = np.concatenate(grid_imgs, axis=0)
        
        # Display
        if grid.ndim == 2:
            ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(grid, vmin=0, vmax=1)
        
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=12)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_comparison(
        self,
        real_images: Union[torch.Tensor, np.ndarray],
        fake_images: Union[torch.Tensor, np.ndarray],
        n_samples: int = 8,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Compare real and generated images side by side.
        
        Args:
            real_images: Shape (N, C, H, W) - real images
            fake_images: Shape (M, C, H, W) - generated images
            n_samples: Number of samples to show from each
            title: Overall figure title
            figsize: Figure size
        
        Returns:
            Tuple of (figure, list of axes)
        
        Example:
            >>> visualizer.plot_comparison(real_batch, fake_batch, n_samples=8)
        """
        real = self._to_numpy(real_images)[:n_samples]
        fake = self._to_numpy(fake_images)[:n_samples]
        
        real = self._normalize_images(real)
        fake = self._normalize_images(fake)
        
        fig, axes = plt.subplots(2, n_samples, figsize=figsize or (n_samples * 1.5, 3))
        
        for i in range(n_samples):
            # Real image
            if i < len(real):
                img = real[i]
                if img.shape[0] == 1:
                    axes[0, i].imshow(img[0], cmap='gray', vmin=0, vmax=1)
                else:
                    axes[0, i].imshow(np.transpose(img, (1, 2, 0)), vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Real', fontsize=10)
            
            # Fake image
            if i < len(fake):
                img = fake[i]
                if img.shape[0] == 1:
                    axes[1, i].imshow(img[0], cmap='gray', vmin=0, vmax=1)
                else:
                    axes[1, i].imshow(np.transpose(img, (1, 2, 0)), vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Fake', fontsize=10)
        
        if title:
            fig.suptitle(title, fontsize=12)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_class_samples(
        self,
        images: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        n_per_class: int = 8,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Display samples organized by class.
        
        Each row shows samples from one class.
        
        Args:
            images: Shape (N, C, H, W) - images
            labels: Shape (N,) - class labels
            n_per_class: Number of samples to show per class
            title: Overall figure title
            figsize: Figure size
        
        Returns:
            Tuple of (figure, array of axes)
        """
        imgs = self._to_numpy(images)
        lbls = self._to_numpy(labels).astype(int)
        imgs = self._normalize_images(imgs)
        
        unique_classes = np.unique(lbls)
        n_classes = len(unique_classes)
        
        fig, axes = plt.subplots(n_classes, n_per_class, 
                                 figsize=figsize or (n_per_class * 1.5, n_classes * 1.5))
        
        if n_classes == 1:
            axes = axes[np.newaxis, :]
        
        for row, c in enumerate(unique_classes):
            class_mask = (lbls == c)
            class_imgs = imgs[class_mask][:n_per_class]
            
            for col in range(n_per_class):
                if col < len(class_imgs):
                    img = class_imgs[col]
                    if img.shape[0] == 1:
                        axes[row, col].imshow(img[0], cmap='gray', vmin=0, vmax=1)
                    else:
                        axes[row, col].imshow(np.transpose(img, (1, 2, 0)), vmin=0, vmax=1)
                axes[row, col].axis('off')
                
                if col == 0:
                    color = self.class_colors.get(c, 'black')
                    axes[row, col].set_ylabel(f'Class {c}', fontsize=10, color=color)
        
        if title:
            fig.suptitle(title, fontsize=12)
        
        plt.tight_layout()
        return fig, axes
    
    # =========================================================================
    # Training History Visualization
    # =========================================================================
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None,
        smooth: bool = True,
        smooth_window: int = 5
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Plot training history curves.
        
        Expected history keys:
        - 'd_loss': Discriminator loss
        - 'g_loss': Generator loss
        - 'd_real_acc': D accuracy on real images
        - 'd_fake_acc': D accuracy on fake images
        - Any other metrics
        
        Args:
            history: Dictionary of metric_name -> list of values
            title: Overall figure title
            figsize: Figure size
            smooth: Whether to apply smoothing
            smooth_window: Window size for moving average smoothing
        
        Returns:
            Tuple of (figure, list of axes)
        
        Example:
            >>> history = trainer.fit(generator, discriminator, ...)
            >>> visualizer.plot_training_history(history)
        """
        # Determine subplot layout - filter out empty metrics
        loss_keys = [k for k in history.keys() if 'loss' in k.lower() and len(history[k]) > 0]
        acc_keys = [k for k in history.keys() if 'acc' in k.lower() and len(history[k]) > 0]
        other_keys = [k for k in history.keys() if k not in loss_keys + acc_keys and len(history[k]) > 0]
        
        n_plots = sum([len(loss_keys) > 0, len(acc_keys) > 0, len(other_keys) > 0])
        if n_plots == 0:
            raise ValueError("No recognized metrics in history")
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize or (5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        def smooth_curve(values, window):
            if len(values) < window:
                return values
            kernel = np.ones(window) / window
            return np.convolve(values, kernel, mode='valid')
        
        plot_idx = 0
        
        # Plot losses
        if loss_keys:
            ax = axes[plot_idx]
            for key in loss_keys:
                values = history[key]
                x = np.arange(len(values))
                
                # Plot original (faded)
                ax.plot(x, values, alpha=0.3)
                
                # Plot smoothed
                if smooth and len(values) >= smooth_window:
                    smoothed = smooth_curve(values, smooth_window)
                    x_smooth = np.arange(smooth_window - 1, len(values))
                    ax.plot(x_smooth, smoothed, label=key, linewidth=2)
                else:
                    ax.plot(x, values, label=key, linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot accuracies
        if acc_keys:
            ax = axes[plot_idx]
            for key in acc_keys:
                values = history[key]
                x = np.arange(len(values))
                
                ax.plot(x, values, alpha=0.3)
                if smooth and len(values) >= smooth_window:
                    smoothed = smooth_curve(values, smooth_window)
                    x_smooth = np.arange(smooth_window - 1, len(values))
                    ax.plot(x_smooth, smoothed, label=key, linewidth=2)
                else:
                    ax.plot(x, values, label=key, linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Discriminator Accuracy')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot other metrics
        if other_keys:
            ax = axes[plot_idx]
            for key in other_keys:
                values = history[key]
                x = np.arange(len(values))
                ax.plot(x, values, label=key, linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title('Other Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_sample_progression(
        self,
        samples_list: List[Union[torch.Tensor, np.ndarray]],
        epochs_list: List[int],
        n_samples: int = 8,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Show how generated samples evolve during training.
        
        Each row corresponds to a different epoch.
        
        Args:
            samples_list: List of sample batches (one per epoch)
            epochs_list: List of epoch numbers corresponding to samples
            n_samples: Number of samples to show per epoch
            title: Overall figure title
            figsize: Figure size
        
        Returns:
            Tuple of (figure, array of axes)
        
        Example:
            >>> samples_epochs = []
            >>> for epoch in [1, 10, 50, 100]:
            ...     z = latent_gen.sample(64)
            ...     samples_epochs.append((generator(z), epoch))
            >>> samples, epochs = zip(*samples_epochs)
            >>> visualizer.plot_sample_progression(samples, epochs)
        """
        n_epochs = len(samples_list)
        
        fig, axes = plt.subplots(n_epochs, n_samples, 
                                 figsize=figsize or (n_samples * 1.5, n_epochs * 1.5))
        
        if n_epochs == 1:
            axes = axes[np.newaxis, :]
        
        for row, (samples, epoch) in enumerate(zip(samples_list, epochs_list)):
            imgs = self._to_numpy(samples)[:n_samples]
            imgs = self._normalize_images(imgs)
            
            for col in range(n_samples):
                if col < len(imgs):
                    img = imgs[col]
                    if img.shape[0] == 1:
                        axes[row, col].imshow(img[0], cmap='gray', vmin=0, vmax=1)
                    else:
                        axes[row, col].imshow(np.transpose(img, (1, 2, 0)), vmin=0, vmax=1)
                axes[row, col].axis('off')
                
                if col == 0:
                    axes[row, col].set_ylabel(f'Epoch {epoch}', fontsize=10)
        
        if title:
            fig.suptitle(title or 'Sample Progression', fontsize=12)
        
        plt.tight_layout()
        return fig, axes
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def save_figure(
        self,
        fig: plt.Figure,
        path: str,
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight'
    ) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure to save
            path: Output file path (supports png, pdf, svg, etc.)
            dpi: Resolution (uses default if None)
            bbox_inches: Bounding box option for tight layout
        """
        fig.savefig(path, dpi=dpi or self.dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to: {path}")
    
    def close_all(self) -> None:
        """Close all open matplotlib figures."""
        plt.close('all')
    
    # =========================================================================
    # Quality Evaluation Methods
    # =========================================================================
    
    def evaluate_sample_quality(
        self,
        fake_images: Union[torch.Tensor, np.ndarray],
        real_images: Optional[Union[torch.Tensor, np.ndarray]] = None,
        n_samples: int = 100,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the quality and diversity of generated samples.
        
        Computes multiple metrics to assess:
        1. Diversity (to detect mode collapse)
        2. Sharpness (to detect blur)
        3. Intra-class variance (another diversity metric)
        4. Statistical similarity to real data (if provided)
        
        Args:
            fake_images: Generated images (N, C, H, W), at least n_samples
            real_images: Real images for comparison (optional)
            n_samples: Number of samples to use for evaluation
            verbose: Whether to print results
        
        Returns:
            Dictionary of metric_name -> value
            
        Metrics:
            - diversity_score: Average pairwise MSE (higher = more diverse)
            - sharpness_score: Average gradient magnitude (higher = sharper)
            - intra_std: Pixel-wise std across samples (higher = more diverse)
            - real_fake_distance: L1 distance between distributions (lower = more similar, only if real_images provided)
        
        Example:
            >>> fake = generator(latent_generator.sample(class_label=1, batch_size=200))
            >>> metrics = visualizer.evaluate_sample_quality(fake, real_minority_samples)
            >>> print(f"Diversity: {metrics['diversity_score']:.4f}")
        """
        # Convert to numpy
        fake = self._to_numpy(fake_images)[:n_samples]
        
        if fake.shape[0] < n_samples:
            print(f"Warning: Only {fake.shape[0]} samples available, using all")
            n_samples = fake.shape[0]
        
        metrics = {}
        
        # =====================================================================
        # 1. Diversity Score (pairwise MSE)
        # =====================================================================
        # Compute MSE between all pairs of generated samples
        # High diversity means samples are different from each other
        fake_flat = fake.reshape(n_samples, -1)  # (N, C*H*W)
        
        # Efficient pairwise distance computation
        pairwise_mse = []
        for i in range(min(50, n_samples)):  # Sample pairs to avoid O(N^2) explosion
            idx = np.random.randint(0, n_samples, size=2)
            while idx[0] == idx[1]:
                idx = np.random.randint(0, n_samples, size=2)
            mse = np.mean((fake_flat[idx[0]] - fake_flat[idx[1]]) ** 2)
            pairwise_mse.append(mse)
        
        metrics['diversity_score'] = float(np.mean(pairwise_mse))
        
        # =====================================================================
        # 2. Sharpness Score (gradient magnitude)
        # =====================================================================
        # Compute image gradients using simple differences
        # Sharp images have high gradient magnitudes
        gradients = []
        for img in fake:
            if img.shape[0] == 1:
                img = img[0]  # Grayscale: (H, W)
            else:
                img = img.mean(axis=0)  # RGB: average channels -> (H, W)
            
            # Compute gradients in x and y directions
            grad_x = np.abs(np.diff(img, axis=1))  # (H, W-1)
            grad_y = np.abs(np.diff(img, axis=0))  # (H-1, W)
            
            # Average gradient magnitude
            grad_mag = (grad_x.mean() + grad_y.mean()) / 2
            gradients.append(grad_mag)
        
        metrics['sharpness_score'] = float(np.mean(gradients))
        
        # =====================================================================
        # 3. Intra-class Variance (pixel-wise std)
        # =====================================================================
        # Compute std across all samples at each pixel
        # High std means samples vary more (more diverse)
        pixel_std = np.std(fake, axis=0)  # (C, H, W)
        metrics['intra_std'] = float(np.mean(pixel_std))
        
        # =====================================================================
        # 4. Statistical Similarity to Real Data
        # =====================================================================
        if real_images is not None:
            real = self._to_numpy(real_images)[:n_samples]
            
            # Compute mean and std of real vs fake
            fake_mean = np.mean(fake)
            fake_std = np.std(fake)
            real_mean = np.mean(real)
            real_std = np.std(real)
            
            # L1 distance between distributions
            mean_diff = np.abs(fake_mean - real_mean)
            std_diff = np.abs(fake_std - real_std)
            metrics['real_fake_distance'] = float(mean_diff + std_diff)
            
            metrics['fake_mean'] = float(fake_mean)
            metrics['fake_std'] = float(fake_std)
            metrics['real_mean'] = float(real_mean)
            metrics['real_std'] = float(real_std)
        
        # Print results
        if verbose:
            print("\n" + "=" * 70)
            print("Sample Quality Evaluation")
            print("=" * 70)
            print(f"  Diversity Score (pairwise MSE):  {metrics['diversity_score']:.6f}")
            print(f"    → Higher is better (more diverse samples)")
            print(f"  Sharpness Score (gradient mag):  {metrics['sharpness_score']:.6f}")
            print(f"    → Higher is better (sharper images)")
            print(f"  Intra-class Std (pixel-wise):    {metrics['intra_std']:.6f}")
            print(f"    → Higher is better (more variance)")
            
            if 'real_fake_distance' in metrics:
                print(f"\n  Statistical Similarity:")
                print(f"    Real mean: {metrics['real_mean']:.6f}, std: {metrics['real_std']:.6f}")
                print(f"    Fake mean: {metrics['fake_mean']:.6f}, std: {metrics['fake_std']:.6f}")
                print(f"    Distance:  {metrics['real_fake_distance']:.6f}")
                print(f"    → Lower is better (more similar to real)")
            
            print("=" * 70)
            
            # Interpretation guidance
            print("\n💡 Interpretation Guide:")
            print("  • Diversity < 0.01: Likely mode collapse")
            print("  • Sharpness < 0.05: Images may be blurry")
            print("  • Intra-std < 0.1: Low variance, check for mode collapse")
            print("=" * 70)
        
        return metrics
    
    def compare_quality_over_epochs(
        self,
        generator: torch.nn.Module,
        latent_generator,
        real_images: Optional[Union[torch.Tensor, np.ndarray]],
        epochs: List[int],
        checkpoint_paths: List[str],
        n_samples: int = 100,
        class_label: int = 1,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, List[float]]]:
        """
        Compare sample quality across different training epochs.
        
        Loads generator checkpoints and evaluates quality metrics.
        Useful for determining the best epoch to stop training.
        
        Args:
            generator: Generator model (will be loaded with checkpoints)
            latent_generator: For sampling latent vectors
            real_images: Real images for comparison
            epochs: List of epoch numbers
            checkpoint_paths: Paths to generator checkpoints for each epoch
            n_samples: Number of samples to generate per epoch
            class_label: Which class to generate
            figsize: Figure size
        
        Returns:
            Tuple of (figure, axes, metrics_dict)
            metrics_dict: {metric_name: [values_per_epoch]}
        
        Example:
            >>> epochs = [10, 30, 50, 100, 150]
            >>> paths = [f'saved/gen_epoch_{e}.pth' for e in epochs]
            >>> fig, ax, metrics = visualizer.compare_quality_over_epochs(
            ...     generator, latent_gen, real_data, epochs, paths)
        """
        device = next(generator.parameters()).device
        all_metrics = {'diversity_score': [], 'sharpness_score': [], 
                      'intra_std': [], 'real_fake_distance': []}
        
        print(f"Comparing quality across {len(epochs)} epochs...")
        
        for epoch, path in zip(epochs, checkpoint_paths):
            print(f"\nEvaluating epoch {epoch}...")
            
            # Load checkpoint
            generator.load_state_dict(torch.load(path, map_location=device))
            generator.eval()
            
            # Generate samples
            with torch.no_grad():
                z = latent_generator.sample(class_label=class_label, batch_size=n_samples)
                fake = generator(z)
            
            # Evaluate quality
            metrics = self.evaluate_sample_quality(fake, real_images, 
                                                   n_samples=n_samples, verbose=False)
            
            for key in all_metrics.keys():
                if key in metrics:
                    all_metrics[key].append(metrics[key])
                else:
                    all_metrics[key].append(None)
        
        # Plot metrics over epochs
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        metric_names = ['diversity_score', 'sharpness_score', 'intra_std', 'real_fake_distance']
        titles = ['Diversity Score\n(higher = better)', 
                 'Sharpness Score\n(higher = better)',
                 'Intra-class Std\n(higher = better)', 
                 'Real-Fake Distance\n(lower = better)']
        
        for i, (metric, title) in enumerate(zip(metric_names, titles)):
            values = all_metrics[metric]
            if any(v is not None for v in values):
                axes[i].plot(epochs, values, marker='o', linewidth=2, markersize=8)
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Value')
                axes[i].set_title(title)
                axes[i].grid(True, alpha=0.3)
                
                # Highlight best epoch
                if metric == 'real_fake_distance':
                    best_idx = np.nanargmin(values)  # Lower is better
                else:
                    best_idx = np.nanargmax(values)  # Higher is better
                axes[i].scatter([epochs[best_idx]], [values[best_idx]], 
                               color='red', s=200, zorder=5, marker='*',
                               label=f'Best: epoch {epochs[best_idx]}')
                axes[i].legend()
        
        plt.tight_layout()
        return fig, axes, all_metrics


# =============================================================================
# Standalone Functions (for quick use without class instantiation)
# =============================================================================

def show_samples(
    images: Union[torch.Tensor, np.ndarray],
    nrow: int = 8,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Quick function to display a grid of images.
    
    Args:
        images: Shape (N, C, H, W) - batch of images
        nrow: Number of images per row
        title: Plot title
        figsize: Figure size
    
    Returns:
        Tuple of (figure, axes)
    """
    viz = BAGANVisualizer()
    return viz.plot_samples_grid(images, nrow=nrow, title=title, figsize=figsize)


def show_latent_space(
    latent_vectors: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    method: str = 'pca',
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Quick function to visualize latent space.
    
    Args:
        latent_vectors: Shape (N, latent_dim)
        labels: Shape (N,)
        method: 'pca' or 'tsne'
        figsize: Figure size
    
    Returns:
        Tuple of (figure, axes)
    """
    viz = BAGANVisualizer()
    return viz.plot_latent_space(latent_vectors, labels, method=method, figsize=figsize)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BAGAN Visualization Module")
    print("=" * 60)
    
    # Test with dummy data (3 classes for N-class generality)
    print("\n[Test 1] Creating visualizer with 3-class palette...")
    viz = BAGANVisualizer()
    print(f"  Colors: {{k: v for k, v in viz.class_colors.items()}}")
    assert 0 in viz.class_colors and 1 in viz.class_colors
    assert 'fake' in viz.class_colors
    assert viz.class_colors[0] == '#1f77b4', "Class 0 should be blue"
    assert viz.class_colors[1] == '#d62728', "Class 1 should be red"
    assert viz.class_colors['fake'] == '#7f7f7f', "Fake should be gray"
    print("  ✓ BAGANVisualizer created with N-class palette")
    
    # Test _build_default_colors with various N
    print("\n[Test 2] Testing _build_default_colors for N=2,3,5...")
    for n in [2, 3, 5]:
        colors = BAGANVisualizer._build_default_colors(n)
        assert len(colors) == n + 1, f"Expected {n+1} entries (N classes + fake), got {len(colors)}"
        assert colors[0] == '#1f77b4', f"Class 0 should always be blue for N={n}"
        assert colors[1] == '#d62728', f"Class 1 should always be red for N={n}"
        assert colors['fake'] == '#7f7f7f', f"Fake should always be gray for N={n}"
        print(f"  ✓ N={n}: {colors}")
    
    # Test sample grid
    print("\n[Test 3] Testing sample grid with random images...")
    dummy_images = torch.randn(16, 1, 64, 64)
    fig, ax = viz.plot_samples_grid(dummy_images, nrow=4, title="Random Noise (Test)")
    plt.close(fig)
    print("  ✓ plot_samples_grid works")
    
    # Test latent space plot with 3 classes
    print("\n[Test 4] Testing latent space visualization (3 classes)...")
    dummy_latents = torch.randn(150, 128)
    dummy_labels = torch.randint(0, 3, (150,))  # 3 classes
    fig, ax = viz.plot_latent_space(dummy_latents, dummy_labels)
    plt.close(fig)
    print("  ✓ plot_latent_space works with 3 classes")
    
    # Test training history plot
    print("\n[Test 5] Testing training history plot...")
    dummy_history = {
        'd_loss': [1.0 - 0.01 * i + 0.1 * np.random.randn() for i in range(50)],
        'g_loss': [1.5 - 0.02 * i + 0.15 * np.random.randn() for i in range(50)],
        'd_real_acc': [0.5 + 0.008 * i + 0.05 * np.random.randn() for i in range(50)],
        'd_fake_acc': [0.5 + 0.009 * i + 0.05 * np.random.randn() for i in range(50)],
    }
    fig, axes = viz.plot_training_history(dummy_history)
    plt.close(fig)
    print("  ✓ plot_training_history works")
    
    # Test comparison plot
    print("\n[Test 6] Testing real vs fake comparison...")
    real_imgs = torch.randn(8, 1, 64, 64)
    fake_imgs = torch.randn(8, 1, 64, 64)
    fig, axes = viz.plot_comparison(real_imgs, fake_imgs, n_samples=8)
    plt.close(fig)
    print("  ✓ plot_comparison works")
    
    # Test class samples with 3 classes
    print("\n[Test 7] Testing class samples display (3 classes)...")
    class_imgs = torch.randn(30, 1, 64, 64)
    class_lbls = torch.cat([torch.full((10,), c) for c in range(3)]).long()
    fig, axes = viz.plot_class_samples(class_imgs, class_lbls, n_per_class=5, title="3-Class Test")
    plt.close(fig)
    print("  ✓ plot_class_samples works with 3 classes")
    
    print("\n" + "=" * 60)
    print("All visualization tests passed!")
    print("=" * 60)
