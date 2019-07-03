
import os, sys, numpy as np, skimage, argparse, warnings
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel

### Visualization helpers ###

# Helper function for displays
def imdisplay(inim, title, colorbar=False, cmap=None):
    fig, ax = plt.subplots()
    imm = ax.imshow(inim) if cmap is None else ax.imshow(inim, cmap=cmap)
    plt.title(title)
    if colorbar: fig.colorbar(imm)

### Patch extraction helpers ###

def patches_over_channels(img, patch_size, window_step, return_meta=True):
    P = np.array([ 
            # view_as_windows(np.ascontiguousarray(img[:,:,k]),
                            # (local_patch_size, local_patch_size), step=wstep) 
            patches_per_channel(img[:,:,k], patch_size, window_step)
            for k in range(3) 
        ])
    if return_meta: return P, P.shape, patch_size**2
    return P

def patches_per_channel(channel, patch_size, window_step):
    # H x W x Px x Py = n_patches_vert x n_patches_horz x patch_H x patch_W
    return view_as_windows(
                np.ascontiguousarray( channel ), 
                (patch_size, patch_size), 
                step=window_step) 

def vectorize_single_masked_patch(patches, mask, i, j, wt):
    mask_patch = mask[i,j,:,:]
    if 0 in mask_patch: return None
    unfolded_patch = patches[:,i,j,:,:].reshape(wt * 3)
    return unfolded_patch

def vectorize_masked_patches(patches, mask, H, W):
    wt = mask.shape[2] * mask.shape[3] # patch/window total size
    P = [ vectorize_single_masked_patch(patches, mask, i, j, wt) 
          for i in range(H) for j in range(W) ]
    return np.array([vp for vp in P if not vp is None])

### Image processing helpers ###

def generate_gradient_magnitude_image(img, divider=2, to_ubyte=False):
    '''
    Generates the per-channel L2 gradient magnitude image of the input `img`.
    Each edge pixel value is divided by `divider` (enforce output in [0,1] + numerical stability).
    If `to_ubyte` is specified True, the output is converted to the ubyte numpy dtype.
    '''
    @adapt_rgb(each_channel)
    def cgrad_x(img):
        return filters.scharr_v(img)
    @adapt_rgb(each_channel)
    def cgrad_y(img):
        return filters.scharr_h(img)
    Ig_x, Ig_y = cgrad_x(img), cgrad_y(img)
    Ig_x_sq, Ig_y_sq = Ig_x * Ig_x, Ig_y * Ig_y
    gradient_img = np.sqrt(Ig_x_sq + Ig_y_sq) / divider # to remain in [0,1]
    # print('GradImg min/max: %.2f/%.2f' % (gradient_img.min(), gradient_img.max()))
    if to_ubyte:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gradient_img = skimage.img_as_ubyte(gradient_img)
    return gradient_img













