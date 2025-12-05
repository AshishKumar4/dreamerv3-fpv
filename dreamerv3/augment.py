"""Physics-safe image augmentation for world model pretraining."""

import jax
import jax.numpy as jnp


def augment_image(image, key, brightness=0.15, contrast=0.15, noise_std=0.015):
    k1, k2, k3 = jax.random.split(key, 3)

    bright_delta = jax.random.uniform(k1, (), minval=-brightness, maxval=brightness)
    image = image + bright_delta

    contrast_factor = jax.random.uniform(k2, (), minval=1-contrast, maxval=1+contrast)
    mean = jnp.mean(image, axis=(-3, -2, -1), keepdims=True)
    image = (image - mean) * contrast_factor + mean

    noise = jax.random.normal(k3, image.shape) * noise_std
    image = image + noise

    return jnp.clip(image, 0.0, 1.0)


def augment_batch(data, key, config):
    if not config.enabled:
        return data

    image = data['image'].astype(jnp.float32) / 255.0
    image = augment_image(
        image, key,
        brightness=config.brightness,
        contrast=config.contrast,
        noise_std=config.noise_std,
    )
    return {**data, 'image': (image * 255).astype(jnp.uint8)}
