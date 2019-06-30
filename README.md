# Img-Complexity

This repo contains implementations of several simple measures of image complexity. 

## Requirements: 

Requires: scikit-image, numpy, matplotlib, and scipy.
An easy way to obtain them is via the conda environment specified in `env.yaml`.

Create environment: `conda env create -f env.yaml`.

Activate environment: `source activate imcom` or `conda activate imcom`.
(On Windows, just `activate imcom`).

## Usage:

Simply run `python img_complexity.py <input>`, where `<input>` can be a folder of images or a single image.

Run `python img_complexity.py --help` to print detailed usage help from the script.

#### Visualizations

You can display intermediate results/images for computations that support it.
For instance, `--show_local_ents` displays a color map with local estimated patch entropies over the image.

#### Gradient Image

The metrics can either be computed on an input image $`I`$ or on the per-channel *gradient magnitude image* 
$`I_G=||\nabla I||_2`$ of that input (or both).

Use `--use_grad_only` to use the gradient image and `--use_grad_too` to compute the measures on *both* the original input and its gradient image.

## Complexity Measures

### Pixel-wise discrete entropy

Computes the discrete Shannon entropy over individual pixel values across the global image, averaged over channels.

### Average local pixel-wise entropies

Computes the mean discrete Shannon entropy on local image patches, averaged over channels.

### Frequency-weighted mean Fourier coefficient

Computes the weighted average of the Fourier coefficient values, with weights based on the associated frequency value.

### Average local patch covariances

Computes the mean local patch covariance over the image via 
```math
\mathcal{C}_L(I) = \frac{1}{|P|} \sum_{p\in P} \log\det\left(\widehat{C}(p)\right)
```
where $P$ is the set of patches, $p\in\mathbb{R}^{S_P \times 3}$ is the set of pixel values per patch, and $\widehat{C}$ is the empirical covariance matrix.

### Average Gradient Magnitude

Computes the mean value of the per-channel gradient magnitude over the image.

### Global Patch Covariance


## Acknowledgements

The global patch-based entropy measure utilizes the *Non-parametric Entropy Estimation Toolbox* by Greg Ver Steeg. 
See the [**NPEET** Github Repo](https://github.com/gregversteeg/NPEET) (MIT licensed).
It implements the approach by Kraskov et al, 2004, *Estimating Mutual Information*.


## TODO

- TODO global patch covariance, global patch entropy
- TODO Handle alpha channel
- TODO allow specifying a subset of complexity measure to run



