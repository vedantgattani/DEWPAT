# Img-Complexity: Implementations of several simple measures of image complexity.

## Requirements: 

Requires: scikit-image, numpy, matplotlib, and scipy.
An easy way to obtain them is via the conda environment specified in `env.yaml`.

Create environment: `conda env create -f env.yaml`.

Activate environment: `source activate imcom` or `conda activate imcom`.
(On Windows, just `activate imcom`).

## Usage:

Simply run `python img_complexity.py <input>`, where `<input>` can be a folder of images or a single image.

Run `python img_complexity.py --help` to print detailed usage help from the script.

### Displays

You can display intermediate results/images for computations that support it.
For instance, `--show_local_ents` displays a color map with local estimated patch entropies over the image.

### Gradient Image

The metrics can either be computed on an input image $`I`$ or on the per-channel *gradient magnitude image* 
$`||\nabla I||_2/2`$ of that input (or both).

Use `--use_grad_only` to use the gradient image and `--use_grad_too` to compute the measures on *both* the original input and the gradient image.

## Complexity Measures

### Pixel-wise entropy

### Average local pixel-wise entropies

### Frequency-weighted mean Fourier coefficient

### Average local patch covariances

### Average Gradient Magnitude


## Acknowledgements

The global patch-based entropy measure utilizes the *Non-parametric Entropy Estimation Toolbox* by Greg Ver Steeg. 
See the [**NPEET** Github Repo](https://github.com/gregversteeg/NPEET) (MIT licensed).
It implements the approach by Kraskov et al, 2004, *Estimating Mutual Information*.


## TODO

- TODO global patch covariance, global patch entropy
- TODO Handle alpha channel
- TODO allow specifying a subset of complexity measure to run



