"""WGAN-GP weight transfer (AE encoder → critic, AE decoder → generator).

Mirrors :class:`bagan.utils.WeightTransfer` API so the ablation harness can
call ``WeightTransfer.initialize_gan_from_autoencoder(autoencoder, generator,
discriminator)`` regardless of backend.

* Decoder → Generator transfer is delegated to vanilla BAGAN's transfer
  function (the generator architecture is identical between backends).
* Encoder → Critic transfer is custom: the WGAN-GP critic has no BatchNorm,
  so we only copy ``Conv2d`` weights/biases. Mapping:

  ::

      encoder.conv_layers.0.{weight,bias}  →  critic.features.0.{weight,bias}
      encoder.conv_layers.3.{weight,bias}  →  critic.features.2.{weight,bias}
      encoder.conv_layers.6.{weight,bias}  →  critic.features.4.{weight,bias}
      encoder.conv_layers.9.{weight,bias}  →  critic.features.6.{weight,bias}

  BatchNorm layers (encoder.conv_layers.{1,4,7,10}) are skipped because the
  critic has none. ``critic.critic_head`` and ``critic.aux_head`` retain their
  random initialisation (different output dims from encoder.fc).
"""

from typing import Dict, Any
import torch

from bagan.utils.weight_transfer import WeightTransfer as _BAGANWeightTransfer

print("bagan_gp/utils/weight_transfer.py loaded")


# Indices of Conv2d layers in encoder.conv_layers Sequential.
# Matches the BAGAN encoder layout: Conv-BN-LReLU repeated 4× (last block omits LReLU).
_ENCODER_CONV_INDICES = (0, 3, 6, 9)
# Indices of Conv2d layers in critic.features Sequential.
_CRITIC_CONV_INDICES = (0, 2, 4, 6)


class WeightTransfer:
    """API-parity wrapper around BAGAN's weight transfer for the WGAN-GP critic.

    Provides ``initialize_gan_from_autoencoder`` and the two granular
    ``transfer_*`` static methods so existing notebook cells keep working.
    """

    # ------------------------------------------------------------------ #
    # Decoder → Generator (delegate — generator is unchanged)
    # ------------------------------------------------------------------ #
    @staticmethod
    def transfer_decoder_to_generator(decoder, generator, verbose: bool = True) -> Dict[str, Any]:
        return _BAGANWeightTransfer.transfer_decoder_to_generator(decoder, generator, verbose=verbose)

    # ------------------------------------------------------------------ #
    # Encoder → Critic (custom — no-BN critic with renumbered conv slots)
    # ------------------------------------------------------------------ #
    @staticmethod
    def transfer_encoder_to_discriminator(encoder, discriminator, verbose: bool = True) -> Dict[str, Any]:
        """Copy encoder Conv2d weights/biases into critic.features Conv2d slots.

        Args:
            encoder: A ``BAGANEncoder`` with ``conv_layers`` ``nn.Sequential``.
            discriminator: A ``WGANAuxCriticNoBN`` with ``features`` ``nn.Sequential``.
            verbose: If True, print per-layer copy report.

        Returns:
            ``{'transferred_params', 'transferred_layers', 'skipped_layers'}``.
        """
        if verbose:
            print("\n" + "=" * 70)
            print("Transferring Weights: Encoder → WGAN-GP Critic (no BN)")
            print("=" * 70)

        transferred_params = 0
        transferred_layers: list = []
        skipped_layers: list = []

        enc_layers = encoder.conv_layers
        crit_layers = discriminator.features

        with torch.no_grad():
            for enc_idx, crit_idx in zip(_ENCODER_CONV_INDICES, _CRITIC_CONV_INDICES):
                src = enc_layers[enc_idx]
                dst = crit_layers[crit_idx]

                # Sanity-check both are Conv2d with matching shapes.
                if not (isinstance(src, torch.nn.Conv2d) and isinstance(dst, torch.nn.Conv2d)):
                    skipped_layers.append(f"encoder.conv_layers.{enc_idx} → critic.features.{crit_idx}")
                    if verbose:
                        print(f"  ✗ slot {enc_idx}→{crit_idx}: not Conv2d, skipped")
                    continue
                if src.weight.shape != dst.weight.shape:
                    skipped_layers.append(f"encoder.conv_layers.{enc_idx} → critic.features.{crit_idx}")
                    if verbose:
                        print(f"  ✗ slot {enc_idx}→{crit_idx}: shape mismatch "
                              f"{list(src.weight.shape)} vs {list(dst.weight.shape)}")
                    continue

                dst.weight.copy_(src.weight)
                transferred_params += src.weight.numel()
                if src.bias is not None and dst.bias is not None and src.bias.shape == dst.bias.shape:
                    dst.bias.copy_(src.bias)
                    transferred_params += src.bias.numel()

                transferred_layers.append(f"conv_layers.{enc_idx} → features.{crit_idx}")
                if verbose:
                    print(f"  ✓ encoder.conv_layers.{enc_idx} → critic.features.{crit_idx}  "
                          f"{list(src.weight.shape)}")

        # The critic_head and aux_head are intentionally left at their random
        # initialisation — different output dims from the encoder's fc layer.
        skipped_layers.extend(["critic.critic_head (random init)",
                               "critic.aux_head (random init)"])

        if verbose:
            print("-" * 70)
            print(f"Transfer Complete!")
            print(f"  Parameters transferred: {transferred_params:,}")
            print(f"  Conv slots transferred: {len(transferred_layers)}")
            print(f"  Layers skipped: {len(skipped_layers)} (BN removed; heads random — expected)")
            print("=" * 70)

        return {
            "transferred_params": transferred_params,
            "transferred_layers": transferred_layers,
            "skipped_layers": skipped_layers,
        }

    # ------------------------------------------------------------------ #
    # Combined entry point
    # ------------------------------------------------------------------ #
    @staticmethod
    def initialize_gan_from_autoencoder(autoencoder, generator, discriminator,
                                        verbose: bool = True) -> Dict[str, Any]:
        """Decoder → Generator (full) + Encoder → Critic (conv only).

        Drop-in equivalent of :meth:`bagan.utils.WeightTransfer.initialize_gan_from_autoencoder`.
        """
        if verbose:
            print("\n" + "=" * 70)
            print("BAGAN-GP Weight Initialization: Autoencoder → WGAN-GP")
            print("=" * 70)

        d2g_stats = WeightTransfer.transfer_decoder_to_generator(
            autoencoder.decoder, generator, verbose=verbose,
        )
        e2c_stats = WeightTransfer.transfer_encoder_to_discriminator(
            autoencoder.encoder, discriminator, verbose=verbose,
        )

        total = d2g_stats["transferred_params"] + e2c_stats["transferred_params"]
        if verbose:
            print("\n" + "=" * 70)
            print(f"  Decoder → Generator: {d2g_stats['transferred_params']:,} parameters")
            print(f"  Encoder → Critic:    {e2c_stats['transferred_params']:,} parameters")
            print(f"  Total transferred:   {total:,} parameters")
            print("=" * 70)

        return {
            "decoder_to_generator": d2g_stats,
            "encoder_to_discriminator": e2c_stats,
            "total_params_transferred": total,
        }
