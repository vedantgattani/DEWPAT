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

#### Examples

- Compute all the complexity measures for a single input image, as well as visualizing the contributions of each pixel for the local pixelwise entropy measure:

  `python img_complexity.py eg.png --show_local_covars`

- Compute only the patch-wise differential entropy measure and the weighed Fourier measure, on both the input image and its gradient image, for every image in the input folder:

  `python img_complexity.py eg_folder --diff_shannon_entropy_patches --weighted_fourier --use_grad_too`

- List all command flags and options of the script, along with descriptions of each:

  `python img_complexity.py --help `

## Complexity Measures

We briefly provide a basic description of the measures. 
See the code for precise computational details.
Notation: image $`I`$ with channels $`c\in C`$, set of patches $`P`$ (per channel, $`P_c`$), set of pixels $`V`$ (per channel, $`V_c`$), and empirical covariance matrix $`\widehat{C}`$.

### Pixel-wise discrete entropy

Computes the discrete Shannon entropy over individual pixel values across the image, written
```math
\mathcal{H}_{d,v}(I) = \frac{-1}{|C|} \sum_{c\in C} \sum_{s\in V_c} p(s) \log p(s)
```

### Average local pixel-wise entropies

Computes the mean discrete Shannon entropy within local image patches, via
```math
\mathcal{H}_{d,p}(I) = \frac{-1}{|C|} \sum_{c\in C} \frac{1}{|P_c|} \sum_{\zeta\in P_c} \sum_{v\in \zeta}  p(v) \log p(v)
```

### Frequency-weighted mean Fourier coefficient

Computes the weighted average of the Fourier coefficient values, with weights based on the associated frequency value, such that
```math
\mathcal{F}_w(I) = \frac{1}{Z_\gamma} \sum_{\psi_x,\,\psi_y\in \Psi} \gamma(\psi_x,\psi_y) \mathcal{I}_F(\psi_x,\psi_y),
\;\text{ where}\;
\mathcal{I}_F = \frac{1}{|C|}\sum_{c\in C} \log \left| \mathfrak{F}[c] \right|
\;\text{and}\; 
Z_\gamma = \sum_{\psi_x,\,\psi_y\in \Psi} \gamma(\psi_x,\psi_y),
```
denoting $`\mathfrak{F}[c]`$ as the Fourier transform of the single-channel image $`c`$, $`\Psi`$ as the set of frequency space coordinates, and $`\gamma(x,y) = |x| + |y|`$ as Manhatten distance weights.

### Average local patch covariances

Estimates the mean local intra-patch covariance over the image, written 
```math
\mathcal{C}_L(I) = \frac{1}{|P|} \sum_{p\in P} \log\left( \det\left(\widehat{C}(p)\right) + 1 \right)
```

### Average Gradient Magnitude

Computes the mean value of the per-channel gradient magnitude over the image; i.e., 
```math
\mathcal{G}(I) = \frac{1}{|C|\,|V_c|} \sum_{c\in C} \sum_{s\in V_c} ||\nabla I(s)||_2
```

### Pixel-wise differential entropy

Estimates the differential Shannon entropy of the continuous-space vector-valued pixel distribution of the image.
This can be written
```math
\mathcal{H}_{c,v}(I) = -\int_V p(v) \log p(v)\, dv
```

### Patch-wise global differential entropy

Estimates the differential entropy of the distribution of patches over the images, where each multi-channel patch is unfolded into a single continuous vector, written
```math
\mathcal{H}_{c,p}(I) = -\int_P p(\xi) \log p(\xi)\, d\xi
```

### Global patch covariance

Computes the log-determinant of the global covariance matrix over patches in the image, where again each multi-channel patch is unfolded into a single vector; i.e.,
```math
\mathcal{C}_{G}(I) = \log\left( \det\left( \widehat{C}(P) \right) \right)
```


## Acknowledgements

The differential entropy measures utilize the *Non-parametric Entropy Estimation Toolbox* by Greg Ver Steeg. 
See the [**NPEET** Github Repo](https://github.com/gregversteeg/NPEET) (MIT licensed).
It implements the approach in Kraskov et al, 2004, *Estimating Mutual Information*.

## TODO

- Better modularization and name-spacing, code redundancy, and efficiency/computation sharing
- Measures looking at the distribution of pairwise distances (L2, Wasserstein, etc...) between patches over the image
