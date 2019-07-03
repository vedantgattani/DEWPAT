import os, sys, numpy as np, skimage, argparse, warnings
import matplotlib.pyplot as plt
from scipy import fftpack as fp
from skimage.morphology import disk
from skimage.util import view_as_windows
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage import filters
import entropy_estimators as EE_cont 
from scipy.stats import entropy as scipy_discrete_shannon_entropy 

from utils import *

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
parser.add_argument('--timing', dest='ignore_alpha', action='store_true',
        help='Whether to ignore the alpha mask channel')
# Specifying complexity measures to use
group_c = parser.add_argument_group('Complexities Arguments',
            description=('Controls which complexity measures to utilize. ' + 
                         'By default, computes all complexity measures; ' + 
                         'if any specific ones are given, only those are used.') )
group_c.add_argument('--discrete_global_shannon',   
        dest='discrete_global_shannon', action='store_const', 
        const=0, default=None,
        help='Whether to use the discrete pixel-wise global Shannon entropy measure')
group_c.add_argument('--discrete_local_shannon',    
        dest='discrete_local_shannon', action='store_const', 
        const=1, default=None,
        help='Whether to use the discrete Shannon entropy across image patches measure')
group_c.add_argument('--weighted_fourier',          
        dest='weighted_fourier', action='store_const', 
        const=2, default=None,
        help='Whether to use the frequency-weighted mean Fourier coefficient measure')
group_c.add_argument('--local_covars',              
        dest='local_covars', action='store_const', 
        const=3, default=None,
        help='Whether to use the average log-determinant of covariances across image patches')
group_c.add_argument('--grad_mag',                  
        dest='grad_mag', action='store_const', 
        const=4, default=None,
        help='Whether to use the mean gradient magnitude over the image')
group_c.add_argument('--diff_shannon_entropy',                  
        dest='diff_shannon_entropy', action='store_const', 
        const=5, default=None,
        help='Whether to use the differential Shannon entropy computed over continuous-valued image pixels')
group_c.add_argument('--diff_shannon_entropy_patches',                  
        dest='diff_shannon_entropy_patches', action='store_const', 
        const=6, default=None,
        help='Whether to use the differential Shannon entropy computed over continuous-valued image patches')
group_c.add_argument('--global_patch_covar',                  
        dest='global_patch_covar', action='store_const', 
        const=7, default=None,
        help='Whether to use the global patch-wise log-determinant of the covariance of unfolded patches across the image')
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
S_all = [ 'Pixelwise Shannon Entropy', 'Average Local Entropies',
          'Frequency-weighted Mean Coefficient', 'Local Patch Covariance',
          'Average Gradient Magnitude', 'Pixelwise Differential Entropy',
          'Patchwise Differential Entropy', 'Global Patch Covariance']
# Discern which measures to use
input_vals = [ args.discrete_global_shannon, args.discrete_local_shannon, args.weighted_fourier,
               args.local_covars, args.grad_mag, args.diff_shannon_entropy, args.diff_shannon_entropy_patches,
               args.global_patch_covar ]
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



def compute_complexities(impath,    # Path to input image file
        complexities_to_use,        # List of ints describing which complexity metric to apply
        to_print=False,             # Switch between multi-image CSV mode and single-image printing mode
        ### Measure parameters ###
        local_entropy_disk_size=24, # Patch size for local entropy calculations
        local_patch_size=16,        # Patch size for local covariance calculations
        wstep=2,                    # The step-size (stride) for the local covariance calculation
        transform_diff_ent=True,    # Whether to affinely transform the differential entropy
        affine_param=150,           # Parameter used in affine transform for differential entropy
        diff_ent_patch_size=3,      # Size of patch to unfold for differential entropy calculations
        diff_ent_window_step=2,     # Patch extraction step size for differential entropy estimation
        global_cov_window_size=5,   # Size of patch for global patch covariances 
        global_cov_window_step=2,   # Patch extraction step size for global patch covariance computations
        global_cov_norm_img=False,  # Whether to normalize the image into [0,1] before computing global covariances
        global_cov_aff_trans=True,  # Whether to affinely transform the output logdet covariances
        global_cov_affine_prm=100,  # Parameter used in affine transform for global patch-wise logdet covariances
        ### Visualization Options ###
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

    The affine parameters for the patch-wise differential entropy and covariance control an order-preserving (monotonic)
        transform for the final output. This is simply for aesthetics, to make the values positive and of smaller magnitude.
        This can be turned off via `transform_diff_ent` and `global_cov_aff_trans`.

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

    # Helper for single parameter affine transform
    _oneparam_affine = lambda x, affp: (x + affp) / affp
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
        def discrete_pixelwise_shannon_entropy(img):
            if using_alpha_mask:
                if verbose: print('Computing image entropy (alpha masked)')  
                def masked_discrete_shannon(channel):
                    channel = np.copy(channel).astype(int)
                    _fake_pixel_val = -1000
                    channel[ alpha_mask <= 0 ] = _fake_pixel_val
                    unique_vals, counts = np.unique(channel, return_counts=True)
                    new_counts = [ c for ii, c in enumerate(counts) if not unique_vals[ii] == _fake_pixel_val ]
                    return scipy_discrete_shannon_entropy(new_counts, base=np.e)
                shannon_entropy = np.mean([masked_discrete_shannon(img[:,:,i]) for i in range(3)])
            else:
                if verbose: print('Computing image entropy')
                shannon_entropy = np.mean([skimage.measure.shannon_entropy(img[:,:,i], base=np.e) for i in range(3)])
            return shannon_entropy
        shannon_entropy = discrete_pixelwise_shannon_entropy(img)
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
        # Manhattan weighting from center
        index_grid_cen = np.array([[ np.abs(i-c[0]) + np.abs(j-c[1])
                            for j in range(0,w)] for i in range(0,h)])
        # Normalize weights into [0,1]. Note this doesn't matter because of the later sum normalization.
        index_grid_cen = index_grid_cen / np.max(index_grid_cen) 
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
        # patches = np.array([ view_as_windows(np.ascontiguousarray(img[:,:,k]),
        #                                      (local_patch_size, local_patch_size), step=wstep) 
        #                     for k in range(3) ])
        # ps, wt = patches.shape, local_patch_size**2
        patches, ps, wt = patches_over_channels(img, local_patch_size, wstep)
        # Per patch: (1) extracts window, (2) unfolds it into pixel values, (3) computes the covariance
        if using_alpha_mask:
            # alpha_over_patches = view_as_windows(np.ascontiguousarray( alpha_mask ), 
            #                         (local_patch_size, local_patch_size), step=wstep) # H x W x Px x Py
            alpha_over_patches = patches_per_channel(alpha_mask, local_patch_size, wstep)
            def masked_detcov(i,j):
                mask_patch = alpha_over_patches[i,j,:,:]
                if 0 in mask_patch: return 0 # Patches with masked pixels don't contribute
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
        if verbose: print('Computing average edginess')
        grad_img = generate_gradient_magnitude_image(img)
        if show_gradient_img:
            gi_title = "Mean Gradient Magnitude Image" + (" (2nd order)" if use_gradient_image else "")
            imdisplay(grad_img.mean(axis=2), gi_title, cmap='plasma', colorbar=True)
        if using_alpha_mask:
            grad_img[alpha_mask <= 0] = 0
        average_edginess = np.mean(grad_img)
        add_new(average_edginess, 4)

    #>> Measure 5: Continuous-space Differential Shannon entropy across pixels
    if 5 in complexities_to_use:
        if verbose: print('Computing continuous pixel-wise differential entropy')
        float_img = skimage.img_as_float(img).reshape(h * w, 3) # list of pixel values
        if using_alpha_mask:
            unfolded_mask = alpha_mask.reshape(h * w)
            _str_tmp = "\tMask Sum: " + str(unfolded_mask.sum()) + ", Orig Shape: " + str(float_img.shape)
            float_img = float_img[ unfolded_mask > 0 ]
            _str_tmp += ", Masked Shape: " + str(float_img.shape)
            if verbose: print(_str_tmp)
        # Compute nearest neighbour-based density estimator
        pixelwise_diff_entropy = EE_cont.entropy(float_img, base=np.e)
        if transform_diff_ent: pixelwise_diff_entropy = _oneparam_affine(pixelwise_diff_entropy, affine_param)
        add_new(pixelwise_diff_entropy, 5)

    #>> Measure 6: Continuous-space Differential Shannon entropy over patches
    # Note: this measure is not invariant to rotations of patches
    if 6 in complexities_to_use:
        if verbose: print('Computing continuous patch-wise differential entropy')
        # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
        float_img = skimage.img_as_float(img)
        patches, ps, wt = patches_over_channels(float_img, diff_ent_patch_size, diff_ent_window_step)
        # patches = np.array([ view_as_windows(np.ascontiguousarray(float_img[:,:,k]),
        #                                      (diff_ent_patch_size, diff_ent_patch_size), 
        #                                      step=diff_ent_window_step) 
        #                     for k in range(3) ])
        # ps, wt = patches.shape, diff_ent_patch_size**2
        if verbose: print('\tPatch windows size:', patches.shape)
        # Gathers unfolded patches across the image
        if using_alpha_mask:
            # alpha_over_patches = view_as_windows(np.ascontiguousarray( alpha_mask ), 
            #                         (diff_ent_patch_size, diff_ent_patch_size), 
            #                         step=diff_ent_window_step) # H x W x Px x Py
            alpha_over_patches = patches_per_channel(alpha_mask, diff_ent_patch_size, diff_ent_window_step)
            # def vectorize_patch(i,j):
            #     mask_patch = alpha_over_patches[i,j,:,:]
            #     if 0 in mask_patch: return None
            #     unfolded_patch = patches[:,i,j,:,:].reshape(wt * 3)
            #     return unfolded_patch
            # patch_vectors = [ vectorize_patch(i,j) for i in range(ps[1]) for j in range(ps[2]) ]
            
            # patch_vectors = [ vectorize_patch(i,j) for i in range(ps[1]) for j in range(ps[2]) ]
            # patch_vectors = np.array([vp for vp in patch_vectors if not vp is None])

            patch_vectors = vectorize_masked_patches(patches, alpha_over_patches, ps[1], ps[2])
        else:
            patch_vectors = np.array([ patches[:,i,j,:,:].reshape(wt * 3) for i in range(ps[1]) for j in range(ps[2]) ])
        if verbose: print('\tPatch vectors shape:', patch_vectors.shape)
        # Compute nearest neighbour-based density estimator
        patchwise_diff_entropy = EE_cont.entropy(patch_vectors, base=np.e)
        if transform_diff_ent: patchwise_diff_entropy = _oneparam_affine(patchwise_diff_entropy, affine_param)
        add_new(patchwise_diff_entropy, 6)

    #> Measure 7: Global patch-wise covariance logdet
    # Note: this measure is also sensitive to patch orientation (e.g., rotating a patch will affect it)
    if 7 in complexities_to_use:
        if verbose: print('Computing global patch-wise covariance')
        # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
        # TODO merge the patch extractor code here with the one just above to reduce reduandancy
        float_img = skimage.img_as_float(img) if global_cov_norm_img else img

        # patches = np.array([ view_as_windows(np.ascontiguousarray(float_img[:,:,k]),
        #                                      (global_cov_window_size, global_cov_window_size), 
        #                                      step=global_cov_window_step) 
        #                     for k in range(3) ])
        # ps, wt = patches.shape, global_cov_window_size**2

        patchesgc, ps, wt = patches_over_channels(float_img, global_cov_window_size, global_cov_window_step)

        if verbose: print('\tPatch windows size:', patchesgc.shape)

        # Gathers unfolded patches across the image
        if using_alpha_mask:
            # alpha_over_patches = view_as_windows(np.ascontiguousarray( alpha_mask ), 
            #                         (global_cov_window_size, global_cov_window_size), 
            #                         step=global_cov_window_step) # H x W x Px x Py
            alpha_over_patchesgc = patches_per_channel(alpha_mask, global_cov_window_size, global_cov_window_step)
            # def vectorize_patch(i,j):
            #     mask_patch = alpha_over_patches[i,j,:,:]
            #     if 0 in mask_patch: return None
            #     unfolded_patch = patches[:,i,j,:,:].reshape(wt * 3)
            #     return unfolded_patch

            patch_vectors = vectorize_masked_patches(patchesgc, alpha_over_patchesgc, ps[1], ps[2])

            # patch_vectors = [ vectorize_patch(i,j) for i in range(ps[1]) for j in range(ps[2]) ]
            # patch_vectors = np.array([vp for vp in patch_vectors if not vp is None])

        else:
            patch_vectors = np.array([ patchesgc[:,i,j,:,:].reshape(wt * 3) for i in range(ps[1]) for j in range(ps[2]) ])
        if verbose: print('\tPatch vectors shape:', patch_vectors.shape)
        # Compute covariance matrix of the unfolded vectors
        global_cov = np.cov( patch_vectors.T )
        sign, patchwise_covar_logdet = np.linalg.slogdet(global_cov) 
        if global_cov_aff_trans: patchwise_covar_logdet = _oneparam_affine(patchwise_covar_logdet, global_cov_affine_prm)
        add_new(patchwise_covar_logdet, 7)

    ### </ Finished computing complexity measures /> ###

    # Minor checks
    assert all([v1 == v2 for v1,v2 in zip(S[1:], computed_names)]), 'Mismatch between intended and computed measures'

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
