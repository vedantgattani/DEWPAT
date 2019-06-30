import os, sys, numpy as np, skimage, argparse, warnings
import matplotlib.pyplot as plt
from scipy import fftpack as fp
from skimage.morphology import disk
from skimage.util import view_as_windows
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
import entropy_estimators as EE 

'''
Usage: python img_complexity.py <input>
    where input is either a folder of images or a single image.
Note: operates in RGB color space.
'''

parser = argparse.ArgumentParser(description='Computes some simple measure of image complexity')
parser.add_argument('input', type=str, help='Input: either a folder or an image')
parser.add_argument('--verbose', dest='verbose', action='store_true',
        help='Whether to print verbosely while running')
# Gradient image usage
parser.add_argument('--use_grad_only', dest='use_grad', action='store_true',
        help='Whether to use the gradient of the image instead of the image itself')
parser.add_argument('--use_grad_too', dest='use_grad_too', action='store_true',
        help='If specified, computes complexities of both original and gradient image')
# Display options
parser.add_argument('--show_fourier', dest='show_fourier', action='store_true',
        help='Whether to display the Fourier transformed and weighted image')
parser.add_argument('--show_local_ents', dest='show_locents', action='store_true',
        help='Whether to display the image of local entropies')
parser.add_argument('--show_local_covars', dest='show_local_covars', action='store_true',
        help='Whether to display an image of the local covariances')
parser.add_argument('--show_gradient_img', dest='show_gradient_img', action='store_true',
        help='Whether to display the gradient magnitude image')
parser.add_argument('--show_img', dest='show_img', action='store_true',
        help='Whether to display the input image')
parser.add_argument('--show_all', dest='show_all', action='store_true',
        help='Whether to display all of the above images')
args = parser.parse_args()

# Names of complexity measures
S = ['Image path', 'Pixelwise Shannon entropy', 'Average local entropies',
     'Frequency-weighted mean coefficient', 'Local patch covariance',
     'Average gradient magnitude']

#------------------------------------------------------------------------------#

def generate_gradient_magnitude_image(img, divider=2.0):
    '''
    Generates the per-channel L2 gradient magnitude image of the input `img`.
    Each edge pixel value is divided by `divider`.
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
    return gradient_img

def compute_complexities(impath,
        to_print=False,
        local_entropy_disk_size=24, # Patch size for local entropy calculations
        local_patch_size=30,        # Patch size for local covariance calculations
        wstep=5,                    # The step-size (stride) for the local covariance calculation
        use_gradient_image=False,   # Whether to use channel-wise gradient image instead of the raw input
        show_fourier_image = False, # Whether to display Fourier-based images
        show_locent_image = False,  # Whether to display the local entropies
        show_loccov_image = False,  # Whether to display the local covariances
        show_gradient_img = False,  # Shows 2nd derivatives if use_gradient_image is true
        display_image = False,      # Whether to display the image under consideration
        verbose=False               # Whether to print verbosely when running
    ):
    '''
    Computes pixelwise entropy, average filtered entropy, weighted frequency content,
        mean local covariance logdets, and average gradient magnitude (edginess).

    If use_gradient_image is specified, the entire function is run on the gradient
        magnitude image of the original image, as if it were the input. The average
        gradient measure in this case, is actually the second-order gradient
        of the original image. This applies when show_gradient_img is True as well.

    Notes:
        These measures still take the background pixels into account.
        RGB color space is used, which may not be perceptually ideal.
        These measures may be image resolution or size dependent.
        They can also be affected by the amount of image area taken up by the dewlap,
            or the length of the boundary of the dewlap in the image.
        May want to exclude the background and/or the edges of the dewlap.
        May want to consider measures based on the structure tensor.
        May want to consider whether to log the covariance determinant or not.
    '''
    # Helper function for displays
    def imdisplay(inim, title, colorbar=False, cmap=None):
        fig, ax = plt.subplots()
        imm = ax.imshow(inim) if cmap is None else ax.imshow(inim, cmap=cmap)
        plt.title(title)
        if colorbar: fig.colorbar(imm)
    # Read original image
    if verbose: print('Reading image:', impath)
    img = skimage.io.imread(impath)
    # Switch to channel-wise gradient magnitude image if desired
    if use_gradient_image:
        if verbose: print('Using gradient magnitude image')
        img = generate_gradient_magnitude_image(img) # Overwrite
    if display_image: imdisplay(img,'Image')
    if verbose:
        print('Image Shape:', img.shape)
        print('Channelwise Min/Max')
        for i in range(3):
            print(i, 'Min:', np.min(img[:,:,i]),'| Max:',np.max(img[:,:,i]))
    # Image dimensions and center
    h, w = img.shape[0:2]
    c = h / 2 - 0.5, w / 2 - 0.5

    #####################################
    ### Computing complexity measures ###
    #####################################

    #>> Channel-wise entropy in nats over pixels
    if verbose: print('Computing image entropy')
    shannon_entropy = np.mean([skimage.measure.shannon_entropy(img[:,:,i],base=np.e) for i in range(3)])

    #>> Averaged channel-wise local entropy
    if verbose: print('Computing local entropies')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        le_img = np.array([ skimage.filters.rank.entropy(img[:,:,i], disk(local_entropy_disk_size))
                    for i in range(3) ]).mean(axis=0)
    if show_locent_image: imdisplay(le_img, 'Local Entropies', colorbar=True, cmap='plasma')
    local_entropy = np.mean(le_img)

    #>> High frequency content via weighted average
    # See e.g., https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    if verbose: print('Computing Fourier images')
    fourier_images = [ fp.fft2(img[:,:,find]) for find in range(3) ]
    shifted_fourier_images = np.array([ np.fft.fftshift(fourier_image) for fourier_image in fourier_images ])
    shifted_fourier_logmag_image = np.array([ np.log( np.abs(shifted_fourier_image) )
                                        for shifted_fourier_image in shifted_fourier_images
                                    ]).transpose((1,2,0)) # Same shape as input
    avg_fourier_image = np.mean(shifted_fourier_logmag_image, axis=2)
    # Manhatten weighting from center
    index_grid_cen = np.array([[ np.abs(i-c[0]) + np.abs(j-c[1])
                        for j in range(0,w)] for i in range(0,h)])
    index_grid_cen = index_grid_cen / np.max(index_grid_cen) # Normalize weights into [0,1]
    fourier_reweighted_image = (avg_fourier_image * index_grid_cen) / np.sum(index_grid_cen)
    fourier_weighted_mean_coef = np.sum( fourier_reweighted_image )
    if show_fourier_image:
        imdisplay(avg_fourier_image, 'Fourier Transform', colorbar=True, cmap='viridis')
        imdisplay(index_grid_cen, 'Fourier-space distance weights', colorbar=True, cmap='gray')
        # Avoid loss of phase information in order to view image (but note it's ignored in the metric)
        reweighted_shifted_fourier_img_wp = (np.mean(shifted_fourier_images, axis=0) * index_grid_cen) / np.sum(index_grid_cen)
        real_space_reweighted_img = np.abs( fp.ifft2( np.fft.ifftshift(reweighted_shifted_fourier_img_wp)) )
        imdisplay(real_space_reweighted_img, 'Reweighted real-space image', colorbar=True, cmap='hot')

    #>> Local (intra-)patch covariances
    if verbose: print('Computing local patch covariances')
    patches = np.array([ view_as_windows(np.ascontiguousarray(img[:,:,k]),
                (local_patch_size, local_patch_size), step=wstep) for k in range(3) ])
    ps, wt = patches.shape, local_patch_size**2
    covariance_mat_dets = [[ np.linalg.det(np.cov( patches[:,i,j,:,:].reshape(wt, 3).T ))
                             for j in range(ps[2]) ] for i in range(ps[1]) ]
    local_covar_img = np.log(np.array(covariance_mat_dets) + 1e-3)
    if show_loccov_image: imdisplay(local_covar_img, 'Local Covariances', cmap='viridis', colorbar=True)
    local_covar = np.mean(local_covar_img)

    #>> Average gradient magnitude of the input
    # Note that this computes the second-order derivatives if we're using a gradient image
    grad_img = generate_gradient_magnitude_image(img)
    if show_gradient_img:
        gi_title = "Mean Gradient Magnitude Image" + (" (2nd order)" if use_gradient_image else "")
        imdisplay(grad_img.mean(axis=2), gi_title, cmap='plasma', colorbar=True)
    average_edginess = np.mean(grad_img)

    ### </ Finished computing complexity measures /> ###

    # Overall complexities
    complexities = [shannon_entropy, local_entropy, fourier_weighted_mean_coef, local_covar, average_edginess]

    # Show images if needed
    if show_fourier_image or display_image or show_locent_image or show_loccov_image or show_gradient_img:
        plt.show()

    ### Print (single image case) and return output ###
    if to_print:
        print(impath + (" (Gradient Magnitude)" if use_gradient_image else ""))
        for name, val in zip(S[1:], complexities): print('\t%s: %.4f' % (name,val))
    impath = impath + (" [GradMag]" if use_gradient_image else "")
    return impath, list(map(lambda f: '%.4f' % f, complexities))
    # </ End of single image complexity calculation /> #

#------------------------------------------------------------------------------#

### Arguments based on user input ###
path = args.input # Path to target
args_d = { "verbose"            : args.verbose,
           "use_gradient_image" : args.use_grad,
           "show_fourier_image" : True if args.show_all else args.show_fourier,
           "show_gradient_img"  : True if args.show_all else args.show_gradient_img,
           "show_locent_image"  : True if args.show_all else args.show_locents,
           "show_loccov_image"  : True if args.show_all else args.show_local_covars,
           "display_image"      : True if args.show_all else args.show_img }
grad_and_orig = args.use_grad_too # Preparations if doing both gradient and original image
if grad_and_orig: del args_d['use_gradient_image']

### Case 1: Compute complexities over a folder of images ###
# Meant to output a CSV
if os.path.isdir(path):
    print(','.join(S))
    for f in [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]:
        p, c = compute_complexities(os.path.join(path,f), **args_d)
        print("%s,%s" % (p,",".join(c)))
        if grad_and_orig: # Just did original
            p, c = compute_complexities(os.path.join(path,f), use_gradient_image=True, **args_d)
            print("%s,%s" % (p,",".join(c)))
### Case 2: Compute complexity measure on a single image ###
else: # Single image case
    compute_complexities(path, to_print=True, **args_d)
    if grad_and_orig:
        compute_complexities(path, to_print=True, use_gradient_image=True, **args_d)


#
