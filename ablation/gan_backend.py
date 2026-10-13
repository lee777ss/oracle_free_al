"""Factory for selecting the GAN backend at runtime.

Two backends are supported:

* ``'bagan'``     — vanilla BAGAN (Mariani et al., 2018; (n+1)-way CE discriminator).
* ``'bagan_gp'``  — WGAN-GP critic with auxiliary classifier (Commit B prototype).

Both packages expose the same public class names (``BAGANConfig``,
``BAGANAutoencoder``, ``BAGANGenerator``, ``BAGANDiscriminator``,
``AutoencoderTrainer``, ``ClassConditionalLatentGenerator``, ``GANTrainer``,
``WeightTransfer``, ``BAGANVisualizer``) plus the three discriminator-adapter
helpers (``realism_score``, ``adv_class_logits``, ``discriminator_features``),
so user code can be written once and dispatched at runtime via
:func:`get_gan_modules`.

Typical usage in notebooks::

    from ablation.gan_backend import get_gan_modules
    gan = get_gan_modules(config.gan_backend)
    BAGANConfig = gan.BAGANConfig
    BAGANGenerator = gan.BAGANGenerator
    ...

And in :mod:`ablation.trainer`::

    helpers = get_gan_modules(getattr(config, "gan_backend", "bagan"))
    self._realism_score = helpers.realism_score
"""

from collections import namedtuple

print("ablation/gan_backend.py loaded")


GANModules = namedtuple(
    "GANModules",
    [
        # Config + models
        "BAGANConfig",
        "BAGANAutoencoder",
        "BAGANGenerator",
        "BAGANDiscriminator",
        # Trainers
        "AutoencoderTrainer",
        "ClassConditionalLatentGenerator",
        "GANTrainer",
        # Helpers
        "WeightTransfer",
        "BAGANVisualizer",
        # Discriminator-adapter helpers (backend-specific)
        "realism_score",
        "adv_class_logits",
        "discriminator_features",
    ],
)


_VALID_BACKENDS = ("bagan", "bagan_gp")


def get_gan_modules(backend: str) -> GANModules:
    """Return the public class set for the requested backend.

    Args:
        backend: ``'bagan'`` or ``'bagan_gp'``.

    Returns:
        A :class:`GANModules` namedtuple. Access fields by name, e.g.
        ``gan.BAGANGenerator``, ``gan.realism_score``.

    Raises:
        ValueError: If ``backend`` is not in :data:`_VALID_BACKENDS`.
    """
    if backend == "bagan":
        import bagan as pkg
    elif backend == "bagan_gp":
        import bagan_gp as pkg
    else:
        raise ValueError(
            f"Unknown gan_backend={backend!r}; expected one of {_VALID_BACKENDS}"
        )

    return GANModules(
        BAGANConfig=pkg.BAGANConfig,
        BAGANAutoencoder=pkg.BAGANAutoencoder,
        BAGANGenerator=pkg.BAGANGenerator,
        BAGANDiscriminator=pkg.BAGANDiscriminator,
        AutoencoderTrainer=pkg.AutoencoderTrainer,
        ClassConditionalLatentGenerator=pkg.ClassConditionalLatentGenerator,
        GANTrainer=pkg.GANTrainer,
        WeightTransfer=pkg.WeightTransfer,
        BAGANVisualizer=pkg.BAGANVisualizer,
        realism_score=pkg.realism_score,
        adv_class_logits=pkg.adv_class_logits,
        discriminator_features=pkg.discriminator_features,
    )
