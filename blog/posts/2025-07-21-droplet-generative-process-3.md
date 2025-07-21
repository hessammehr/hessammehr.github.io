# Trialing probabilistic modelling for chemical microscopy (part 3)

Can we used an amortized model to speed up inference in our droplet microscopy model?


```python
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import seaborn as sns
from numpyro import deterministic, plate, sample
from numpyro.handlers import seed, trace, substitute
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from PIL import Image

plt.rcParams['figure.dpi'] = 200

sns.set_theme(context='paper', style='ticks', font='Arial')
```


```python
img = Image.open('data/example.jpg')
img = img.resize((img.width // 4, img.height // 4))
img
```

For simplicity, we'll focus on modeling the H (hue) channel of the image.


```python
img_hsv = np.array(img.convert('HSV')) / 255.0

plt.imshow(img_hsv[..., 0], cmap='gray')
plt.colorbar()
```


```python
class DropletOpticsModel(nn.Module):
    """
    Neural network model for amortized inference of droplet optics.
    
    Takes pixel background values, distance from droplet, droplet radius,
    and droplet composition to predict new pixel values.
    """
    hidden_dims: tuple = (32, 16, 8)
    
    @nn.compact
    def __call__(self, background, dx, dy, radius, composition):
        """
        Args:
            background: (batch, n_channels) - existing/background pixel values
            dx: (batch, 1) - normalized distance from droplet center in x-direction
            dy: (batch, 1) - normalized distance from droplet center in y-direction
            radius: (batch, 1) - droplet radius
            composition: (batch, n_composition_features) - droplet composition vector
        
        Returns:
            new_pixel_value: (batch, n_channels) - predicted new pixel values
        """
        # Concatenate all input features
        x = jnp.concatenate([background, dx, dy, radius, composition], axis=-1)

        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f'hidden_{i}')(x)
            x = nn.LayerNorm(name=f'ln_{i}')(x)
            x = jnn.relu(x)
        
        # Output layer - predict change in pixel values
        delta = nn.Dense(background.shape[-1], name='output')(x)
        
        # Add residual connection and apply sigmoid to keep values in [0, 1]
        new_pixel_value = jnn.sigmoid(background + delta)
        
        return new_pixel_value

# Initialize model
model = DropletOpticsModel()

# Test with dummy data
key = jax.random.PRNGKey(42)
batch_size, n_channels, n_composition = 32, 3, 10

dummy_background = jax.random.uniform(key, (batch_size, n_channels))
dummy_dx = jax.random.uniform(key, (batch_size, 1), minval=-1.0, maxval=1.0)
dummy_dy = jax.random.uniform(key, (batch_size, 1), minval=-1.0, maxval=1.0)
dummy_radius = jax.random.uniform(key, (batch_size, 1)) * 0.1  # Small radii
dummy_composition = jax.random.uniform(key, (batch_size, n_composition))

# Initialize parameters
print(f"{dummy_background.shape=}, {dummy_dx.shape=}, {dummy_dy.shape=}, {dummy_radius.shape=}, {dummy_composition.shape=}")
params = model.init(key, dummy_background, dummy_dx, dummy_dy, dummy_radius, dummy_composition)

# Test forward pass
output = model.apply(params, dummy_background, dummy_dx, dummy_dy, dummy_radius, dummy_composition)
print(f"Input background shape: {dummy_background.shape}")
print(f"Output pixel values shape: {output.shape}")
print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

# Print model summary
print(f"\nModel summary:")
print(model.tabulate(key, dummy_background, dummy_dx, dummy_dy, dummy_radius, dummy_composition))
```


```python
dummy_background.shape
```


```python
def tree_to_dists(tree, path=''):
    if isinstance(tree, dict):
        return {k: tree_to_dists(v, path + '/' + k) for k, v in tree.items()}
    else:
        # print(f"Sampling {path} with shape {tree.shape}")
        return sample(path, dist.Normal().expand(tree.shape))
```


```python
def model_with_nn(img, n_droplets, types=10):
    h, w, n_channels = img.shape
    
    # Create coordinate grids
    y_coords, x_coords = jnp.mgrid[:h, :w]
    
    # Sample background per channel
    with plate("channels", n_channels):
        bg = sample("bg", dist.Uniform(0, 1))

    # Sample droplet parameters
    with plate("droplets", n_droplets):
        x = sample("x", dist.Uniform(0, w))
        y = sample("y", dist.Uniform(0, h))
        r = sample("r", dist.LogNormal(0, 0.5))
        with plate("types", types):
            composition = sample("composition", dist.Uniform(0, 1))

    model = DropletOpticsModel()

    # Initialize background image
    prediction = jnp.broadcast_to(bg, (h, w, n_channels))
    nn_params = model.init(key, 
                           jnp.zeros((h * w, n_channels)), 
                           jnp.zeros((h * w, 1)),
                           jnp.zeros((h * w, 1)),
                           jnp.zeros((h * w, 1)),
                           jnp.zeros((h * w, types)))    
    nn_params = tree_to_dists(nn_params)
    # For each droplet, compute its effect using the neural network
    for i in range(n_droplets):
        # Calculate relative distances from droplet center
        dx = (x_coords - x[i]) / r[i]  # Normalized by radius
        dy = (y_coords - y[i]) / r[i]  # Normalized by radius
        
        # Flatten spatial dimensions for neural network processing
        dx_flat = dx.flatten()[:, None]
        dy_flat = dy.flatten()[:, None]
        r_flat = jnp.full((h * w, 1), r[i])
        
        # Repeat background and composition for all pixels
        bg_flat = jnp.broadcast_to(prediction.reshape(-1, n_channels), (h * w, n_channels))
        comp_flat = jnp.broadcast_to(composition[:, i], (h * w, types))
        
        # print(f"{bg_flat.shape=}, {dx_flat.shape=}, {dy_flat.shape=}, {r_flat.shape=}, {comp_flat.shape=}")
        # Apply neural network to get new pixel values
        new_pixels = model.apply(nn_params, bg_flat, dx_flat, dy_flat, r_flat, comp_flat)
        
        # Reshape back to image dimensions
        new_pixels = new_pixels.reshape(h, w, n_channels)
        
        # Update prediction (could be additive or replacement - using replacement here)
        prediction = new_pixels
    
    prediction = jnp.clip(prediction, 0, 1)
    prediction = deterministic('prediction', prediction)
    diff = deterministic('diff', img - prediction)
    sample('obs', dist.Normal(scale=0.05), obs=diff)
    return nn_params
```


```python
# Test the integrated model
test_img = jnp.ones((50, 50, 3)) * 0.5  # Small test image

# Use the neural network parameters from the model we initialized earlier
tr = trace(seed(model_with_nn, 0)).get_trace(test_img, 3, types=5)
nn_params = seed(model_with_nn, 0)(test_img, 3, types=5)
{k: v['value'].shape for k, v in tr.items() if 'value' in v}
```


```python
mcmc = MCMC( NUTS(model_with_nn), num_warmup=1000, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), test_img, 3, types=5)
```


```python
tr = trace(seed(model, 0)).get_trace(np.array(img), 15)
{k: v['value'].shape for k, v in tr.items() if 'value' in v}
```


```python
guide = AutoNormal(model)
svi = SVI(model, guide, Adam(0.01), Trace_ELBO())

svi_result = svi.run(jax.random.PRNGKey(0), 100000, img.width, img.height, 2000, img_hsv[..., 0])
samples_svi = guide.sample_posterior(jax.random.PRNGKey(0), svi_result.params, sample_shape=(100,))
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(svi_result.losses)
```


```python
plt.imshow(samples_svi['img'].mean(axis=0), cmap='gray')
plt.colorbar()
```

Looks quite good!


```python
plt.imshow(img_hsv[:, :, 0]/255.0, cmap='gray')
plt.colorbar()
plt.scatter(samples_svi['x'][:100] * img_hsv.shape[1], samples_svi['y'][:100] * img_hsv.shape[0], s=4, alpha=0.01, c='red', marker='x')
```

Most droplets are now detected — very nice!

Let's have a look at the inferred droplet masks:
