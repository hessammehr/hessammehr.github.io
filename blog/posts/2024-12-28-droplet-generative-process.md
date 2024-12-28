```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import seaborn as sns
from numpyro import deterministic, plate, sample
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam
from PIL import Image

sns.set_theme('notebook', 'ticks', font='Arial')

plt.rcParams['figure.dpi'] = 200
```

# Trialing generative processes for chemical microscopy

Is it possible to use a generative process to model microscope images like this (and is it worth the effort?)


```python
img = Image.open('data/example.jpg')
img = img.resize((img.width // 2, img.height // 2))
```

Try a couple of different colour spaces in case something interesting stands out.


```python
from skimage import color

img_array = np.array(img)
lab_img = color.rgb2lab(img_array)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (ax, title) in enumerate(zip(axes, ['L channel', 'a channel', 'b channel'])):
    im = ax.imshow(lab_img[:,:,i], cmap='gray')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

plt.tight_layout()
```


    
![png](2024-12-28-droplet-generative-process_files/2024-12-28-droplet-generative-process_4_0.png)
    



```python
hsv_img = color.rgb2hsv(img_array)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (ax, title) in enumerate(zip(axes, ['H channel', 'S channel', 'V channel'])):
    im = ax.imshow(hsv_img[:,:,i], cmap='gray')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

plt.tight_layout()
```


    
![png](2024-12-28-droplet-generative-process_files/2024-12-28-droplet-generative-process_5_0.png)
    


First approach, fixed number of droplets; model centres and radii


```python
def model1(w, h, n_droplets):
    with plate("droplets", n_droplets):
        x = sample("x", dist.Uniform(0, w))
        y = sample("y", dist.Uniform(0, h))
        r = sample("r", dist.LogNormal(1.5, 0.75))


mcmc = MCMC(NUTS(model1), num_warmup=1000, num_samples=100)
mcmc.run(jax.random.PRNGKey(0), w=img.width, h=img.height, n_droplets=100)
samples = mcmc.get_samples()

fig, ax = plt.subplots()
ax.imshow(np.ones_like(np.array(img)) * 255, cmap="gray")
for sample_no in range(5):
    for i in range(100):
        circle = plt.Circle(
            (samples["x"][sample_no][i], samples["y"][sample_no][i]),
            samples["r"][sample_no][i],
            color=plt.cm.tab10(sample_no),
        )

        ax.add_artist(circle)
```

    sample: 100%|██████████| 1100/1100 [00:09<00:00, 111.22it/s, 15 steps of size 3.08e-01. acc. prob=0.84]



    
![png](2024-12-28-droplet-generative-process_files/2024-12-28-droplet-generative-process_7_1.png)
    



```python
def model2(w, h, n_droplets):
    # Sample droplet locations and sizes
    with plate("droplets", n_droplets):
        x = sample("x", dist.Uniform(0, w))
        y = sample("y", dist.Uniform(0, h))
        r = sample("r", dist.LogNormal(1.5, 0.75))
        
        # Sample HSV values for each droplet
        h_val = sample("h", dist.Uniform(0, 1))
        s_val = sample("s", dist.Beta(2, 2))
        v_val = sample("v", dist.Beta(5, 2))  # Biased towards brighter values

mcmc = MCMC(NUTS(model2), num_warmup=1000, num_samples=100)
mcmc.run(jax.random.PRNGKey(0), w=img.width, h=img.height, n_droplets=100)
samples = mcmc.get_samples()
samples = {k: np.array(v) for k, v in samples.items()}

# Visualize with HSV colors
fig, ax = plt.subplots()
ax.imshow(np.ones_like(np.array(img)) * 255, cmap="gray")
for i in range(100):
    circle = plt.Circle(
        (samples["x"][0][i], samples["y"][0][i]),
        samples["r"][0][i],
        color=color.hsv2rgb(
            np.array(
                [
                    samples["h"][0][i],
                    samples["s"][0][i],
                    samples["v"][0][i],
                ]
            )
        ),
    )
    ax.add_artist(circle)
```

    sample: 100%|██████████| 1100/1100 [00:14<00:00, 73.33it/s, 15 steps of size 2.52e-01. acc. prob=0.86] 



    
![png](2024-12-28-droplet-generative-process_files/2024-12-28-droplet-generative-process_8_1.png)
    



```python
def model2(w, h, n_droplets):
    # Sample droplet locations and sizes
    with plate("droplets", n_droplets):
        x = sample("x", dist.Uniform(0, w))
        y = sample("y", dist.Uniform(0, h))
        r = sample("r", dist.LogNormal(1.5, 0.75))
        
        with plate("pixels", w * h):
            x_dist = jnp.abs(x - jnp.arange(w)[:, None])
            y_dist = jnp.abs(y - jnp.arange(h)[:, None])
            distance = jnp.sqrt(x_dist ** 2 + y_dist[:, None] ** 2)
            val = deterministic('val', jnp.sum(jnp.exp(-distance ** 2 / (2 * r ** 2)), axis=-1))


mcmc = MCMC(NUTS(model2), num_warmup=500, num_samples=10)
mcmc.run(jax.random.PRNGKey(0), w=img.width, h=img.height, n_droplets=100)
samples = mcmc.get_samples()
samples = {k: np.array(v) for k, v in samples.items()}

# show the first 3 samples
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(1 - samples['val'][i], cmap='gray')
```

    sample: 100%|██████████| 510/510 [00:06<00:00, 78.05it/s, 15 steps of size 3.10e-01. acc. prob=0.85] 



    
![png](2024-12-28-droplet-generative-process_files/2024-12-28-droplet-generative-process_9_1.png)
    


Not a bad generative process to start with. Now let's just fit the hue channel ...


```python
def model3(w, h, n_droplets, channel):
    # Sample droplet locations and sizes
    bg = sample("bg", dist.Uniform(0, 1))
    with plate("droplets", n_droplets):
        x = sample("x", dist.Uniform(0, 1))*w
        y = sample("y", dist.Uniform(0, 1))*h
        r = sample("r", dist.LogNormal(1.5, 0.75))
        amplitude = sample("amplitude", dist.HalfNormal())
        
        x_dist = jnp.abs(x - jnp.arange(w)[:, None])
        y_dist = jnp.abs(y - jnp.arange(h)[:, None])
        distance = jnp.sqrt(x_dist ** 2 + y_dist[:, None] ** 2)
        val = deterministic('val', bg + jnp.sum(amplitude[None, None, :] * jnp.exp(-distance ** 2 / (2 * r ** 2)), axis=-1))
        diff = deterministic('diff', val - channel)
    sample('obs', dist.Normal(0, 0.1), obs=val - channel)

```

Fitting, instead of sampling


```python
guide = AutoNormal(model3)
svi = SVI(model3, guide, Adam(0.02), Trace_ELBO())
svi_result = None
while True:
    svi_result = svi.run(jax.random.PRNGKey(0), 1000, img.width, img.height, 500, hsv_img[:,:,0], init_state=svi_result and svi_result.state)
    if svi_result.losses[0] - svi_result.losses[-1] < 1000:
        break
```

    100%|██████████| 1000/1000 [00:17<00:00, 55.98it/s, init loss: 11135004.0000, avg. loss [951-1000]: 933170.7500]
    100%|██████████| 1000/1000 [00:17<00:00, 57.68it/s, init loss: 930588.8125, avg. loss [951-1000]: 877631.7500]
    100%|██████████| 1000/1000 [00:17<00:00, 57.46it/s, init loss: 878014.4375, avg. loss [951-1000]: 838963.0625]
    100%|██████████| 1000/1000 [00:17<00:00, 57.45it/s, init loss: 838473.2500, avg. loss [951-1000]: 813781.2500]
    100%|██████████| 1000/1000 [00:17<00:00, 55.65it/s, init loss: 813162.3750, avg. loss [951-1000]: 792063.2500]
    100%|██████████| 1000/1000 [00:17<00:00, 57.15it/s, init loss: 790267.4375, avg. loss [951-1000]: 775268.8125]
    100%|██████████| 1000/1000 [00:17<00:00, 56.96it/s, init loss: 774699.1875, avg. loss [951-1000]: 764176.7500]
    100%|██████████| 1000/1000 [00:17<00:00, 57.26it/s, init loss: 764443.5000, avg. loss [951-1000]: 754463.6250]
    100%|██████████| 1000/1000 [00:17<00:00, 56.95it/s, init loss: 754262.7500, avg. loss [951-1000]: 744407.1875]
    100%|██████████| 1000/1000 [00:17<00:00, 57.05it/s, init loss: 743674.8750, avg. loss [951-1000]: 739275.9375]
    100%|██████████| 1000/1000 [00:17<00:00, 57.13it/s, init loss: 739285.9375, avg. loss [951-1000]: 735915.7500]
    100%|██████████| 1000/1000 [00:17<00:00, 57.00it/s, init loss: 735452.8750, avg. loss [951-1000]: 731554.0000]
    100%|██████████| 1000/1000 [00:17<00:00, 57.09it/s, init loss: 730866.3750, avg. loss [951-1000]: 728556.8125]
    100%|██████████| 1000/1000 [00:17<00:00, 57.05it/s, init loss: 728374.0625, avg. loss [951-1000]: 722425.6250]
    100%|██████████| 1000/1000 [00:17<00:00, 57.14it/s, init loss: 721568.3750, avg. loss [951-1000]: 718221.0625]
    100%|██████████| 1000/1000 [00:17<00:00, 57.12it/s, init loss: 718202.4375, avg. loss [951-1000]: 713332.8125]
    100%|██████████| 1000/1000 [00:17<00:00, 57.09it/s, init loss: 713491.5625, avg. loss [951-1000]: 712046.0625]
    100%|██████████| 1000/1000 [00:17<00:00, 57.19it/s, init loss: 711863.6250, avg. loss [951-1000]: 705859.1250]
    100%|██████████| 1000/1000 [00:17<00:00, 57.17it/s, init loss: 705953.6875, avg. loss [951-1000]: 702366.8125]
    100%|██████████| 1000/1000 [00:17<00:00, 57.16it/s, init loss: 702441.9375, avg. loss [951-1000]: 700419.1250]
    100%|██████████| 1000/1000 [00:17<00:00, 57.12it/s, init loss: 700093.9375, avg. loss [951-1000]: 698991.6875]
    100%|██████████| 1000/1000 [00:17<00:00, 57.04it/s, init loss: 699521.0000, avg. loss [951-1000]: 696912.8125]
    100%|██████████| 1000/1000 [00:17<00:00, 57.04it/s, init loss: 696551.1875, avg. loss [951-1000]: 693607.8125]
    100%|██████████| 1000/1000 [00:17<00:00, 57.00it/s, init loss: 693494.9375, avg. loss [951-1000]: 693099.8125]



```python
samples = guide.sample_posterior(jax.random.PRNGKey(0), svi_result.params, sample_shape=(5,))
```


```python
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
fig.colorbar(axes[0].imshow(samples['val'][0], cmap='gray'), ax=axes[0], fraction=0.03, pad=0.04)
axes[0].set_title('Prediction')
fig.colorbar(axes[1].imshow(jnp.abs(samples['diff'][0]), cmap='gray'), ax=axes[1], fraction=0.03, pad=0.04)
axes[1].set_title('Difference')
fig.colorbar(axes[-1].imshow(hsv_img[:,:,0], cmap='gray'), ax=axes[-1], fraction=0.03, pad=0.04)
axes[-1].set_title('Ground truth')
fig.tight_layout()
```


    
![png](2024-12-28-droplet-generative-process_files/2024-12-28-droplet-generative-process_15_0.png)
    


Not a bad start
