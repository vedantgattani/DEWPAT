import os, sys, numpy as np, skimage, argparse, warnings
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from timeit import default_timer as timer

### Visualization helpers ###

# Helper function for displays
def imdisplay(inim, title, colorbar=False, cmap=None):
    fig, ax = plt.subplots()
    imm = ax.imshow(inim) if cmap is None else ax.imshow(inim, cmap=cmap)
    plt.title(title)
    if colorbar: fig.colorbar(imm)

def patch_display(patches, nrows, ncols, show=False, title=None, subtitles=None):
    # patches must be n_patches x H x W x C 
    N = nrows * ncols
    assert N == patches.shape[0] and len(subtitles) == N
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        #img = np.random.randint(10, size=(h,w))
        axi.imshow(patches[i,:,:,:])
        #axi.imshow(img, alpha=0.25)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        if subtitles is None:
        # write row/col indices as axes' title for identification
            axi.set_title( ('%d' % i) + "-R"+str(rowid)+"C"+str(colid))
        else:
            subtitle = subtitles[i]
            axi.set_title( str(subtitle) )
    plt.tight_layout(True)
    if not title is None: fig.canvas.set_window_title(title)
    if show: plt.show()

### Patch extraction helpers ###

def patches_over_channels(img, patch_size, window_step, return_meta=True, floatify=False):
    # C x H x W x Px x Py = num_channels x n_patches_vert x n_patches_horz x patch_H x patch_W
    if floatify: 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = skimage.img_as_float(img)
    P = np.array([ 
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

def vectorize_single_masked_patch_as_list(patches, mask, i, j, wt):
    mask_patch = mask[i,j,:,:]
    if 0 in mask_patch: return None
    listified_patch = patches[:,i,j,:,:].reshape(3, wt).T
    return listified_patch

def vectorize_masked_patches(patches, mask, H, W, as_list=False, flatten=True, remove_none=True):
    wt = mask.shape[2] * mask.shape[3] # patch/window total size
    if flatten:
        if as_list:
            P = [ vectorize_single_masked_patch_as_list(patches, mask, i, j, wt) 
                  for i in range(H) for j in range(W) ]
        else:
            P = [ vectorize_single_masked_patch(patches, mask, i, j, wt) 
                  for i in range(H) for j in range(W) ]
    else:
        if as_list:
            P = [ [ vectorize_single_masked_patch_as_list(patches, mask, i, j, wt) 
                    for i in range(H) ] for j in range(W) ]
        else:
            P = [ [ vectorize_single_masked_patch(patches, mask, i, j, wt) 
                    for i in range(H) ] for j in range(W) ]
    if remove_none:
        return np.array([vp for vp in P if not vp is None])
    else:
        return P

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

def gaussian_blur(image, sigma):
    '''
    Performs Gaussian blurring on the input with standard deviation sigma.
    '''
    return np.clip(filters.gaussian(image, sigma=sigma, multichannel=True), 0.0, 1.0)

def to_perceptual_greyscale(img):
    from skimage.color import rgb2gray
    new_img = dupe_channel( rgb2gray(img) )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_img = skimage.img_as_ubyte( new_img )
    return new_img

def to_avg_greyscale(img):
    new_img = dupe_channel( np.clip(np.mean(img / 255, axis=-1), 0, 1) )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_img = skimage.img_as_ubyte( new_img )
    return new_img

def conv_to_ubyte(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_ubyte( img )
    return img

### Timing helpers ###

# Note: calling this as a decorator actually runs it, so that 
# replacer is the actual decorator used.
def timing_decorator(do_timing):
    def decorator(wrapped_func):
        if do_timing:
            def replacement(*args, **kwargs):
                start = timer()
                output = wrapped_func(*args, **kwargs)
                end = timer()
                return output, (end - start)
            return replacement
        else:
            return wrapped_func
    return decorator

### Array Helpers ###

def dupe_channel(c):
    return combine_channels([c, c, c])

def combine_channels(channels):
    return np.stack( channels, axis = -1 )




#
