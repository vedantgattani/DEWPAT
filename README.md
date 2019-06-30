# Img-Complexity

This repo contains implementations of several simple measures of image complexity,
including ones based on frequency content, information entropy, and spatial derivatives.

## Requirements

Requires: scikit-image, numpy, matplotlib, and scipy.
An easy way to obtain them is via the conda environment specified in `env.yaml`.

Create environment: `conda env create -f env.yaml`.

Activate environment: `source activate imcom` or `conda activate imcom`.
(On Windows, just `activate imcom`).

## Usage

Simply run `python img_complexity.py <input>`, where `<input>` can be a folder of images or a single image.

Run `python img_complexity.py --help` to print detailed usage help from the script.

#### Visualizations

You can display intermediate results/images for computations that support it.
For instance, `--show_local_ents` displays a color map with local estimated patch entropies over the image.

#### Gradient Image

The metrics can either be computed on an input image $`I`$ or on the per-channel *gradient magnitude image* 
$`I_G=||\nabla I||_2`$ of that input (or both).

Use `--use_grad_only` to use the gradient image and `--use_grad_too` to compute the measures on *both* the original input and its gradient image.

#### Alpha Channel Masking

Most measures are able to account for the presence of an alpha channel mask. 
For instance, in most patch-based estimators, any patch with a masked pixel is ignored. 
The alpha channel can be ignored with the flag `--ignore_alpha`.

## Complexity Measures

We briefly provide a basic description of the measures. 
See the code for precise computational details.
Notation: image $`I`$ with channels $`c\in C`$, set of patches $`P`$ (per channel, $`P_c`$), set of pixels $`V`$ (per channel, $`V_c`$), and empirical covariance matrix $`\widehat{C}`$.

### Pixel-wise discrete entropy

Computes the discrete Shannon entropy over individual pixel values across the image, written
```math
\mathcal{H}_{d,v}(I) = \frac{-1}{|C|} \sum_{c\in C} \sum_{v\in V_c} p(v) \log p(v)
```

### Average local pixel-wise entropies

Computes the mean discrete Shannon entropy within local image patches, via
```math
\mathcal{H}_{d,p}(I) = \frac{-1}{|C|} \sum_{c\in C} \frac{1}{|P_c|} \sum_{\zeta\in P_c} \sum_{v\in \zeta}  p(v) \log p(v)
```

### Frequency-weighted mean Fourier coefficient

Computes the weighted average of the Fourier coefficient values, with weights based on the associated frequency value.

### Average local patch covariances

Computes the mean local intra-patch covariance over the image, written 
```math
\mathcal{C}_L(I) = \frac{1}{|P|} \sum_{p\in P} \log\det\left(\widehat{C}(p)\right)
```

### Average Gradient Magnitude

Computes the mean value of the per-channel gradient magnitude over the image; i.e., 
```math
\mathcal{G}(I) = \frac{1}{|C|\,|V_c|} \sum_{c\in C} \sum_{v\in V_c} ||\nabla I(v)||_2
```

### Pixel-wise differential entropy

Computes the differential Shannon entropy of the continuous-space vector-valued pixel distribution of the image.
This can be written
```math
\mathcal{H}_{c,v}(I) = -\int_V p(v) \log p(v)\, dv
```

### Patch-wise global differential entropy

Computes the differential entropy of the distribution of patches over the images, where each multi-channel patch is unfolded into a single continuous vector.
This can be written
```math
\mathcal{H}_{c,p}(I) = -\int_P p(\xi) \log p(\xi)\, d\xi
```

### Global patch covariance


## Acknowledgements

The differential entropy measures utilize the *Non-parametric Entropy Estimation Toolbox* by Greg Ver Steeg. 
See the [**NPEET** Github Repo](https://github.com/gregversteeg/NPEET) (MIT licensed).
It implements the approach in Kraskov et al, 2004, *Estimating Mutual Information*.

