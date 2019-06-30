import os, sys, numpy as np, skimage, argparse, warnings
import matplotlib.pyplot as plt
from scipy import fftpack as fp
from skimage.morphology import disk
from skimage.util import view_as_windows
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
import entropy_estimators as EE 
from scipy.stats import entropy as scipy_discrete_shannon_entropy 


'''
Usage: python img_complexity.py <input>
    where input is either a folder of images or a single image.
By default, computes all complexity measures; if any specific ones are given, only those are used.
Note: operates in RGB color space.
'''

overall_desc = 'Computes some simple measure of image complexity.'
parser = argparse.ArgumentParser(description=overall_desc)
# Input image
parser.add_argument('input', type=str, help='Input: either a folder or an image')
# Meta-arguments
parser.add_argument('--verbose', dest='verbose', action='store_true',
        help='Whether to print verbosely while running')
parser.add_argument('--ignore_alpha', dest='ignore_alpha', action='store_true',
        help='Whether to ignore the alpha mask channel')
# Specifying complexity measures to use
group_c = parser.add_argument_group('Complexities Arguments',
            description=('Controls which complexity measures to utilize. ' + 
                         'By default, computes all complexity measures; ' + 
                         'if any specific ones are given, only those are used.') )
group_c.add_argument('--discrete_global_shannon',   
        dest='discrete_global_shannon', action='store_const', const=0, default=None,
        help='Whether to use the discrete pixel-wise global Shannon entropy measure')
group_c.add_argument('--discrete_local_shannon',    
        dest='discrete_local_shannon', action='store_const', const=1, default=None,
        help='Whether to use the discrete Shannon entropy across image patches measure')
group_c.add_argument('--weighted_fourier',          
        dest='weighted_fourier', action='store_const', const=2, default=None,
        help='Whether to use the frequency-weighted mean Fourier coefficient measure')
group_c.add_argument('--local_covars',              
        dest='local_covars', action='store_const', const=3, default=None,
        help='Whether to use the average log-determinant of covariances across image patches')
group_c.add_argument('--grad_mag',                  
        dest='grad_mag', action='store_const', const=4, default=None,
        help='Whether to use the mean gradient magnitude over the image')
# Display options
group_v = parser.add_argument_group('Visualization Arguments',
            description='Options for viewing intermediate computations.')
group_v.add_argument('--show_fourier',      
        dest='show_fourier', action='store_true',
        help='Whether to display the Fourier transformed and weighted image')
group_v.add_argument('--show_local_ents',   
        dest='show_locents', action='store_true',
        help='Whether to display the image of local entropies')
group_v.add_argument('--show_local_covars', 
        dest='show_local_covars', action='store_true',
        help='Whether to display an image of the local covariances')
group_v.add_argument('--show_gradient_img', 
        dest='show_gradient_img', action='store_true',
        help='Whether to display the gradient magnitude image')
group_v.add_argument('--show_img',          
        dest='show_img', action='store_true',
        help='Whether to display the input image')
group_v.add_argument('--show_all',          
        dest='show_all', action='store_true',
        help='Whether to display all of the above images')
# Gradient image usage
group_g = parser.add_argument_group('Gradient Image Input Arguments')
group_g.add_argument('--use_grad_only',     
        dest='use_grad', action='store_true',
        help='Whether to use the gradient of the image instead of the image itself')
group_g.add_argument('--use_grad_too',      
        dest='use_grad_too', action='store_true',
        help='If specified, computes complexities of both the original and the gradient image')
#> Final parsing <#
args = parser.parse_args()

# Names of complexity measures
S_all = [ 'Pixelwise Shannon entropy', 'Average local entropies',
          'Frequency-weighted mean coefficient', 'Local patch covariance',
          'Average gradient magnitude']
# Discern which measures to use
input_vals = [ args.discrete_global_shannon, args.discrete_local_shannon, args.weighted_fourier,
               args.local_covars, args.grad_mag ]
if all(map(lambda k: k is None, input_vals)):
    if args.verbose: print('Using all complexity measures')
    S = S_all
    input_vals = list(range(len(input_vals)))
else:
    S = [ S_all[i] for i in range(len(input_vals)) if i in input_vals ]
    if args.verbose: print('Using:', ", ".join(S))
S = ['Image path'] + S
complexities_to_use = [ val for val in input_vals if not val is None ]

#------------------------------------------------------------------------------#

def generate_gradient_magnitude_image(img, divider=2.0, to_ubyte=False):
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
    if to_ubyte:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gradient_img = skimage.img_as_ubyte(gradient_img)
    return gradient_img

def compute_complexities(impath,    # Path to input image file
        complexities_to_use,        # List of ints describing which complexity metric to apply
        to_print=False,             # Switch between multi-image CSV mode and single-image printing mode
        local_entropy_disk_size=24, # Patch size for local entropy calculations
        local_patch_size=16,        # Patch size for local covariance calculations
        wstep=2,                    # The step-size (stride) for the local covariance calculation
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
    # Handle alpha transparency
    n_channels = img.shape[2]
    alpha_channel = img[:,:,3] if n_channels == 4 else None
    using_alpha_mask = not (alpha_channel is None)
    if using_alpha_mask:
        img = img[:,:,0:3]
        alpha_mask = np.copy(alpha_channel).astype(int)
        alpha_mask[ alpha_mask <= 0 ] = 0
        alpha_mask[ alpha_mask  > 0 ] = 1
        if False: # Display mask
            imdisplay(alpha_mask, 'alpha_mask', colorbar=True, cmap='plasma')
            plt.show()
    if args.ignore_alpha:
        using_alpha_mask = False
        alpha_mask, alpha_channel = None, None
    # Switch to channel-wise gradient magnitude image if desired
    if use_gradient_image:
        if verbose: print('Using gradient magnitude image')
        img = generate_gradient_magnitude_image(img, to_ubyte=True) # Overwrite
    # Perform masking directly on the image (to destroy details hidden behind the alpha channel)
    if using_alpha_mask:
        img[ alpha_mask <= 0 ] = 0
    if display_image: imdisplay(img, 'Image')
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

    complexities, computed_names = [], []
    def add_new(val, ind):
        complexities.append(val)
        computed_names.append(S_all[ind])

    #>> Measure 0: Channel-wise entropy in nats over pixels
    if 0 in complexities_to_use:
        if using_alpha_mask:
            if verbose: print('Computing image entropy (alpha masked)')  
            def masked_discrete_shannon(channel):
                channel = np.copy(channel).astype(int)
                _fake_pixel_val = -1000
                channel[ alpha_mask <= 0 ] = _fake_pixel_val
                unique_vals, counts = np.unique(channel, return_counts=True)
                # print('Vals\n', unique_vals)
                # print('Counts\n', counts)
                new_counts = [ c for ii, c in enumerate(counts) if not unique_vals[ii] == _fake_pixel_val ]
                # print('New Counts\n', new_counts)
                return scipy_discrete_shannon_entropy(new_counts, base=np.e)
            shannon_entropy = np.mean([masked_discrete_shannon(img[:,:,i]) for i in range(3)])
        else:
            if verbose: print('Computing image entropy')
            shannon_entropy = np.mean([skimage.measure.shannon_entropy(img[:,:,i], base=np.e) for i in range(3)])
        add_new(shannon_entropy, 0)

    #>> Measure 1: Averaged channel-wise local entropy
    if 1 in complexities_to_use:
        if verbose: print('Computing local discrete entropies')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            current_mask_local_ent = alpha_mask if using_alpha_mask else None
            le_img = np.array([ skimage.filters.rank.entropy(img[:,:,i], 
                                                             disk(local_entropy_disk_size),
                                                             mask=current_mask_local_ent)
                        for i in range(3) ]).mean(axis=0)
        if show_locent_image: imdisplay(le_img, 'Local Entropies', colorbar=True, cmap='plasma')
        local_entropy = np.mean(le_img)
        add_new(local_entropy, 1)

    #>> Measure 2: High frequency content via weighted average
    # See e.g., https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
    # Note: masks, even if present, are not used here, since we would be getting irregular domain harmonics rather than Fourier coefs
    # This does mean, however, that background pixels do still participate in the measure (e.g., a white dewlap on a white bg, will
    # incur different frequency effects than on a black bg). 
    if 2 in complexities_to_use:
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
        add_new(fourier_weighted_mean_coef, 2)

    #>> Measure 3: Local (intra-)patch covariances
    if 3 in complexities_to_use:
        if verbose: print('Computing local patch covariances')
        # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
        patches = np.array([ view_as_windows(np.ascontiguousarray(img[:,:,k]),
                                             (local_patch_size, local_patch_size), step=wstep) 
                            for k in range(3) ])
        ps, wt = patches.shape, local_patch_size**2
        # Per patch: (1) extracts window, (2) unfolds it into pixel values, (3) computes the covariance
        if using_alpha_mask:
            alpha_over_patches = view_as_windows(np.ascontiguousarray( alpha_mask ), 
                                    (local_patch_size, local_patch_size), step=wstep) # H x W x Px x Py
            def masked_detcov(i,j):
                mask_patch = alpha_over_patches[i,j,:,:]
                if 0 in mask_patch: return 0
                unfolded_patch = patches[:,i,j,:,:].reshape(wt, 3).T
                return np.linalg.det(np.cov( unfolded_patch ))
            covariance_mat_dets = [ [ masked_detcov(i,j) for j in range(ps[2]) ] for i in range(ps[1]) ]
        else:
            covariance_mat_dets = [[ np.linalg.det(np.cov( patches[:,i,j,:,:].reshape(wt, 3).T ))
                                 for j in range(ps[2]) ] for i in range(ps[1]) ]
        _num_corrector = 1.0
        local_covar_img = np.log(np.array(covariance_mat_dets) + _num_corrector)
        if show_loccov_image: imdisplay(local_covar_img, 'Local Covariances', cmap='viridis', colorbar=True)
        local_covar = np.mean(local_covar_img)
        add_new(local_covar, 3)

    #>> Measure 4: Average gradient magnitude of the input
    # Note: this computes the second-order derivatives if we're using a gradient image
    # Note: even if we mask the gradient image, the gradient on the boundary will still be high
    if 4 in complexities_to_use:
        grad_img = generate_gradient_magnitude_image(img)
        if show_gradient_img:
            gi_title = "Mean Gradient Magnitude Image" + (" (2nd order)" if use_gradient_image else "")
            imdisplay(grad_img.mean(axis=2), gi_title, cmap='plasma', colorbar=True)
        if using_alpha_mask:
            grad_img[alpha_mask <= 0] = 0
        average_edginess = np.mean(grad_img)
        add_new(average_edginess, 4)

    ### </ Finished computing complexity measures /> ###

    # Minor checks
    assert all([v1 == v2 for v1,v2 in zip(S[1:], computed_names)]), 'Mismatch between intended and computed measures'

    # Overall complexities
    #complexities = [shannon_entropy, local_entropy, fourier_weighted_mean_coef, local_covar, average_edginess]

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
        p, c = compute_complexities(os.path.join(path,f), complexities_to_use, **args_d)
        print("%s,%s" % (p,",".join(c)))
        if grad_and_orig: # Just did original
            p, c = compute_complexities(os.path.join(path,f), complexities_to_use, 
                                        use_gradient_image=True, **args_d)
            print("%s,%s" % (p,",".join(c)))
### Case 2: Compute complexity measure on a single image ###
else: # Single image case
    compute_complexities(path, complexities_to_use, to_print=True, **args_d)
    if grad_and_orig:
        compute_complexities(path, complexities_to_use, to_print=True, use_gradient_image=True, **args_d)


#
