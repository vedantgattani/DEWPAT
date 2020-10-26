import numpy as np, skimage, warnings
from .entropy_estimators import entropy
from scipy import fftpack as fp
from skimage.morphology import disk
from scipy.stats import entropy as scipy_discrete_shannon_entropy
from ..utils import *


#>> Measure 0: Channel-wise entropy in nats over pixels
@timing_decorator()
def discrete_pixelwise_shannon_entropy(img, alpha_mask=None, verbose=False, timing=False):
    if verbose: print('Computing image entropy ' + ('(alpha masked)' if not alpha_mask is None else ''))  
    if not alpha_mask is None:
        def masked_discrete_shannon(channel, first):
            channel = np.copy(channel).astype(int)
            _fake_pixel_val, alpha_mask_threshold = -1000, 0
            channel[ alpha_mask <= alpha_mask_threshold ] = _fake_pixel_val
            unique_vals, counts = np.unique(channel, return_counts=True)
            new_counts = [ c for ii, c in enumerate(counts) if not unique_vals[ii] == _fake_pixel_val ]
            if first and verbose: print('\tN_pixels before & after masking:', sum(counts), '->', sum(new_counts))
            return scipy_discrete_shannon_entropy(new_counts, base=np.e)
        shannon_entropy = np.mean([masked_discrete_shannon(img[:,:,i], i==0) for i in range(3)])
    else:
        shannon_entropy = np.mean([skimage.measure.shannon_entropy(img[:,:,i], base=np.e) for i in range(3)])
    return shannon_entropy


#>> Measure 1: Averaged channel-wise local entropy
@timing_decorator()
def channelwise_local_entropies(img, alpha_mask=None, show_locent_image=False, local_entropy_disk_size=24,
                                verbose=False, timing=False):
    if verbose: print('Computing local discrete entropies')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        current_mask_local_ent = alpha_mask
        le_img = np.array([ skimage.filters.rank.entropy(img[:,:,i], 
                                                            disk(local_entropy_disk_size),
                                                            mask=current_mask_local_ent)
                    for i in range(3) ]).mean(axis=0)
    if show_locent_image: 
        imdisplay(le_img, 'Local Entropies', colorbar=True, cmap='plasma', mask=alpha_mask)
    return np.mean(le_img)


#>> Measure 2: High frequency content via weighted average
# Note: masks, even if present, are not used here, since we would be getting irregular domain harmonics rather than Fourier coefs
# This does mean, however, that background pixels do still participate in the measure (e.g., a white dewlap on a white bg, will
# incur different frequency effects than on a black bg). 
@timing_decorator()
def mean_weighted_fourier_coef(img, show_fourier_image=False, verbose=False, timing=False):
    h, w = img.shape[0:2]
    c = h / 2 - 0.5, w / 2 - 0.5
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
        mean_shifted_fourier_images = np.mean(shifted_fourier_images, axis=0)
        reweighted_shifted_fourier_img_wp = (mean_shifted_fourier_images * index_grid_cen) / np.sum(index_grid_cen)
        real_space_reweighted_img = np.abs( fp.ifft2( np.fft.ifftshift(reweighted_shifted_fourier_img_wp)) )
        imdisplay(real_space_reweighted_img, 'Reweighted real-space image', colorbar=True, cmap='hot')
    return fourier_weighted_mean_coef


#>> Measure 3: Local (intra-)patch covariances
# Closely related to the distribution of L_2 distance between patches
# Note: vectorize over the patches? use slogdet
@timing_decorator()
def local_patch_covariance(img, alpha_mask=None, local_patch_size=20, local_covar_wstep=5, show_loccov_image=False,
                           verbose=False, timing=False):
    if verbose: print('Computing local patch covariances')
    is_scalar = len(img) == 2
    # In the greyscale case, use trace instead of determinant
    if is_scalar: scalarize = np.trace
    else:         scalarize = np.linalg.det
    # patches \in C x H x W x Px x Py; ps = patches.shape; wt = window length squared
    patches, ps, wt = patches_over_channels(img, local_patch_size, local_covar_wstep)
    # Per patch: (1) extracts window, (2) unfolds it into pixel values, (3) computes the covariance
    if not alpha_mask is None:
        alpha_over_patches = patches_per_channel(alpha_mask, local_patch_size, local_covar_wstep)
        def masked_detcov(i,j):
            mask_patch = alpha_over_patches[i,j,:,:]
            if 0 in mask_patch: return 0 # Patches with masked pixels don't contribute
            unfolded_patch = patches[:,i,j,:,:].reshape(3, wt)
            return scalarize(np.cov( unfolded_patch ))
        if verbose: 
            print('\tPatches Size: %s (Mask Size: %s)' % 
                    (str(patches.shape),str(alpha_over_patches.shape)) )
        covariance_mat_dets = [ [ masked_detcov(i,j) for j in range(ps[2]) ] for i in range(ps[1]) ]
    else:
        if verbose: print('\tPatches Size:', patches.shape)
        #print( "Patches", [ [ patches[:,i,j,:,:].reshape(3,wt) for j in range(ps[2]) ] for i in range(ps[1]) ] )
        #print( "covs", [ [ np.cov( patches[:,i,j,:,:].reshape(3, wt) )
        #                        for j in range(ps[2]) ] for i in range(ps[1]) ] )
        #sys.exit(0)
        covariance_mat_dets = [[ scalarize(np.cov( patches[:,i,j,:,:].reshape(3, wt) ))
                                    for j in range(ps[2]) ] for i in range(ps[1]) ]
    _num_corrector = 1.0
    local_covar_img = np.log(np.array(covariance_mat_dets) + _num_corrector)
    if show_loccov_image:
        h, w, c = img.shape
        resized_local_covar_img = skimage.transform.resize(local_covar_img, output_shape=(h,w), order=3)
        imdisplay(resized_local_covar_img, 'Local Covariances', cmap = 'viridis', 
                    colorbar = True, mask = alpha_mask) 
    return np.mean(local_covar_img)


#>> Measure 4: Average gradient magnitude of the input
# Note: this computes the second-order derivatives if we're using a gradient image
# Note: even if we mask the gradient image, the gradient on the boundary will still be high
@timing_decorator()
def avg_gradient_norm(img, alpha_mask=None, use_gradient_image=False, show_gradient_img=False, verbose=False, timing=False):
    if verbose: print('Computing average edginess')
    grad_img = generate_gradient_magnitude_image(img)
    if show_gradient_img:
        gi_title = "Mean Gradient Magnitude Image" + (" (2nd order)" if use_gradient_image else "")
        imdisplay(grad_img.mean(axis=2), gi_title, cmap='plasma', colorbar=True, mask = alpha_mask)
    if not alpha_mask is None:
        grad_img[alpha_mask <= 0] = 0
    return np.mean(grad_img)


#>> Measure 5: Continuous-space Differential Shannon entropy across pixels
@timing_decorator()
def cont_pixelwise_diff_ent(img, alpha_mask=None, transform_diff_ent=False, affine_param=150, verbose=False, timing=False):
    h, w = img.shape[0:2]
    if verbose: print('Computing continuous pixel-wise differential entropy')
    float_img = skimage.img_as_float(img).reshape(h * w, 3) # list of pixel values
    if not alpha_mask is None:
        unfolded_mask = alpha_mask.reshape(h * w)
        _str_tmp = "\tMask Sum: " + str(unfolded_mask.sum()) + ", Orig Shape: " + str(float_img.shape)
        float_img = float_img[ unfolded_mask > 0 ]
        _str_tmp += ", Masked Shape: " + str(float_img.shape)
        if verbose: print(_str_tmp)
    # Compute nearest neighbour-based density estimator
    pixelwise_diff_entropy = entropy(float_img, base=np.e)
    _oneparam_affine = lambda x, affp: (x + affp) / affp
    if transform_diff_ent: pixelwise_diff_entropy = _oneparam_affine(pixelwise_diff_entropy, affine_param)
    return pixelwise_diff_entropy


#>> Measure 6: Continuous-space Differential Shannon entropy over patches
# Note 1: this measure is not invariant to rotations of patches
# Note 2: for images with large swathes of identical patches, this method can suffer large increases
#   in computational cost (due to KD-tree construction/querying difficulties I suspect).
@timing_decorator()
def patchwise_diff_ent(img, alpha_mask=None, diff_ent_patch_size=3, diff_ent_window_step=2, max_patches_cont_DE=10000,
                       affine_param=150, transform_diff_ent=False, verbose=False, timing=False):
    if verbose: print('Computing continuous patch-wise differential entropy')
    cimg = skimage.img_as_float(img) # + 1e-6 * np.random.randn(*img.shape)
    # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
    patches_dse, ps, wt = patches_over_channels(cimg, diff_ent_patch_size, diff_ent_window_step, floatify=False)
    if verbose: print('\tPatch windows size:', patches_dse.shape)
    # Gathers unfolded patches across the image
    if not alpha_mask is None:
        alpha_over_patches = patches_per_channel(alpha_mask, diff_ent_patch_size, diff_ent_window_step)
        patch_vectors = vectorize_masked_patches(patches_dse, alpha_over_patches, ps[1], ps[2])
    else:
        patch_vectors = np.array([ patches_dse[:,i,j,:,:].reshape(wt * 3) for i in range(ps[1]) for j in range(ps[2]) ])
    # Randomly resample patches to meet maximum number present
    if max_patches_cont_DE < patch_vectors.shape[0]:
        if verbose: print('\tResampling patch vectors (original size: %s)' % str(patch_vectors.shape))
        new_inds = np.random.choice(patch_vectors.shape[0], size=max_patches_cont_DE, replace=False)
        patch_vectors = patch_vectors[new_inds]
    if verbose: print('\tPatch vectors shape:', patch_vectors.shape)
    # Compute nearest neighbour-based density estimator
    patchwise_diff_entropy = entropy(patch_vectors, base=np.e)
    _oneparam_affine=lambda x, affp: (x + affp) / affp
    if transform_diff_ent: patchwise_diff_entropy = _oneparam_affine(patchwise_diff_entropy, affine_param)
    return patchwise_diff_entropy


#>> Measure 7: Global patch-wise covariance logdet
# Note: this measure is also sensitive to patch orientation (e.g., rotating a patch will affect it)
@timing_decorator()
def patchwise_global_covar(img, alpha_mask=None, global_cov_window_size=10, global_cov_window_step=2,
                           global_cov_norm_img=False, global_cov_aff_trans=False,
                           global_cov_affine_prm=100, verbose=False, timing=False):
    if verbose: print('Computing global patch-wise covariance')
    is_scalar = len(img.shape) == 2
    # Patches: channels x patch_index_X x patch_index_Y x coord_in_patch_X x coord_in_patch_Y
    # I.e., C x N_patches_H x N_patches_V x Window_size_H x Window_size_V
    patchesgc, ps, wt = patches_over_channels(img, global_cov_window_size, global_cov_window_step, floatify=global_cov_norm_img)
    if verbose: print('\tPatch windows size:', patchesgc.shape)
    # Gathers unfolded patches across the image
    if not alpha_mask is None:
        alpha_over_patchesgc = patches_per_channel(alpha_mask, global_cov_window_size, global_cov_window_step)
        patch_vectors = vectorize_masked_patches(patchesgc, alpha_over_patchesgc, ps[1], ps[2])
    else:
        patch_vectors = np.array([ patchesgc[:,i,j,:,:].reshape(wt * 3) for i in range(ps[1]) for j in range(ps[2]) ])
    if verbose: print('\tPatch vectors shape:', patch_vectors.shape)
    # Compute covariance matrix of the unfolded vectors
    global_cov = np.cov( patch_vectors.T )
    def _strace(c):
        t = np.trace(c)
        return np.sign(t), t
    scalarizer = (lambda c: np.linalg.slogdet(c)) if not is_scalar else (lambda c: _strace(c))
    sign, patchwise_covar_logdet = scalarizer(global_cov) 
    #patchwise_covar_logdet = np.log( np.linalg.det(global_cov) + 1.0 )
    _oneparam_affine=lambda x, affp: (x + affp) / affp
    if global_cov_aff_trans: patchwise_covar_logdet = _oneparam_affine(patchwise_covar_logdet, global_cov_affine_prm)
    return patchwise_covar_logdet


#> Measure 8: Pairwise patch EMD (i.e., mean pairwise Wasserstein distance over patches)
# TODO bugged? doesnt remove alpha masked patches
@timing_decorator()  
def pairwise_wasserstein_distance(img, use_sinkhorn=True, sinkhorn_gamma=0.25, coordinate_scale=0.2, coordinate_aware=False,
        image_rescaling_factor=1.0, emd_visualize=True, metric='sqeuclidean', alpha_mask=None, emd_window_size=24,
        emd_window_step=16, verbose=False, timing=False):
    import ot
    assert metric in ['euclidean', 'sqeuclidean'], "Underlying metric must be 'euclidean' or 'sqeuclidean'"
    if verbose: print('\tImage dims', img.shape)
    # Resize image
    img = skimage.transform.rescale(img, scale=image_rescaling_factor, anti_aliasing=True, multichannel=True)
    if verbose: print('\tDownscaled image dims:', img.shape)
    if emd_visualize: imdisplay(img, 'Downscaled img')
    # Extract patches
    patches_emd, ps, wt = patches_over_channels(img, emd_window_size, emd_window_step, floatify=True)
    if verbose: print('\tPatches Shape', patches_emd.shape)
    # CASE 1: using alpha mask
    if not alpha_mask is None:
        if emd_visualize: imdisplay( alpha_mask, 'Original Mask', True )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha_mask_ds = skimage.img_as_ubyte(
                                skimage.transform.rescale( 
                                    skimage.img_as_float(skimage.img_as_ubyte(alpha_mask*255)), 
                                    scale=image_rescaling_factor) )
        alpha_mask_ds[ alpha_mask_ds >  1e-8 ] = 1
        alpha_mask_ds[ alpha_mask_ds <= 1e-8 ] = 0
        if emd_visualize: imdisplay(alpha_mask_ds, 'Downscaled mask', True)
        alpha_over_patches_emd = patches_per_channel(alpha_mask_ds, emd_window_size, emd_window_step)
        patch_lists = vectorize_masked_patches(patches_emd, alpha_over_patches_emd, ps[1], ps[2], as_list=True)
    # CASE 2: no alpha channel
    else:
        patch_lists = np.array([ patches_emd[:,i,j,:,:].reshape(3, wt).T 
                                    for i in range(ps[1]) for j in range(ps[2]) ])
    if verbose: print('\tPatch lists shape:', patch_lists.shape)
    # Visualize (random or first few) patches
    if emd_visualize:  
        nr, nc, _random_vis = 5, 4, True
        n_to_vis = nr * nc
        if n_to_vis > patch_lists.shape[0]: 
            ris = np.random.choice(patch_lists.shape[0], size=n_to_vis, replace=True)
        elif _random_vis: 
            ris = np.random.choice(patch_lists.shape[0], size=n_to_vis, replace=False)
        else:             
            ris = np.array( range(n_to_vis) )
        # Index into list (N_patches x unfolded_window_size x 3)
        patches_to_vis = patch_lists[ris, :, :].reshape(n_to_vis, emd_window_size, emd_window_size, 3)
        patch_display(patches_to_vis, nr, nc, show=False, title='EMD Patches', subtitles=ris)
    # Append coordinates if desired
    if coordinate_aware:
        if verbose: print('\tUsing coordinate-aware calculations')
        n_patches = patch_lists.shape[0]
        # Refold vectorized patches back into windows
        refolded_patches = patch_lists.reshape(n_patches, emd_window_size, emd_window_size, 3)
        # Linear walk from 0 to 1
        lin_walk = np.linspace(0.0, 1.0, emd_window_size)
        # Expand linear walk to normalized patch coordinates, and rescale spatial values
        patch_of_coords = np.transpose( np.array( np.meshgrid(lin_walk, lin_walk) ), (1,2,0) ) * coordinate_scale
        # Duplicate patches for each image patch
        coord_patches = np.repeat( patch_of_coords[ np.newaxis, :, : ], n_patches, axis=0 )
        # Concatenate image patch pixel values with coordinate values
        patch_lists = np.concatenate( (coord_patches, refolded_patches), axis=3 ).reshape(n_patches, wt, 5)
        if verbose: print('\tPatch sizes with appended coords:', patch_lists.shape)
    ## Choose solver ##
    if use_sinkhorn: # Entropy-regularized approximate solver
        solver = lambda a,b,M: ot.sinkhorn2(a,b,M,sinkhorn_gamma)
    else: # Exact linear programming solver
        solver = ot.emd2
    n = patch_lists.shape[0]
    ## Function for computing EMD between two image patches ##
    def pair_emd(xs, xt, ind, should_print):
        if verbose and should_print: print('\tComputing EMD set %d/%d' % (ind+1,n))
        # Distance matrix
        M = ot.dist(xs, xt, metric=metric) #+ 1e-7 # Prevent M.max() from being zero
        max_d = M.max()
        if max_d < 1e-6: return 0.0
        M /= max_d
        # Uniform empirical delta density weightings
        a, b = ot.unif(wt), ot.unif(wt)
        # Compute EMD
        emd_final = solver(a,b,M)
        if type(emd_final) is float: return emd_final
        else: return emd_final[0]
    # Compute EMDs
    if verbose: print('\tComputing pairwise EMDs')
    emds = np.array([ pair_emd(patch_lists[i], patch_lists[j], i, j == i+1) 
                        for i in range(n) for j in range(i+1,n) ])
    if verbose: 
        _vss = ( len(emds), np.min(emds), np.max(emds), np.std(emds) )
        print('\tNum/Min/Max/Stdev EMD values: %d/%.2f/%.2f/%.2f' % _vss)
    # Combine EMDs into a single measure via averaging
    D_w = emds.mean() 
    return D_w


# Measure 9: mean-matching pairwise distances
@timing_decorator()
def patchwise_mean_dist(img, mask, pw_mnt_dist_nonOL_WS=[3,4], show_pw_mnt_ptchs=False, verbose=False, timing=False):
    return pairwise_moment_distances(img, mask, block_cuts=pw_mnt_dist_nonOL_WS,
            gamma_mu_weight=1.0,
            gamma_cov_weight=0.0, 
            display_intermeds=show_pw_mnt_ptchs, verbose=verbose)


# Measure 10: moment-matching pairwise distances
@timing_decorator()
def patchwise_moment_dist(img, mask, pw_mnt_dist_nonOL_WS=[3,4], gamma_mu_weight=1.0, show_pw_mnt_ptchs=False,
                          gamma_cov_weight=1.0, verbose=False, timing=False):
    return pairwise_moment_distances(img, mask, block_cuts=pw_mnt_dist_nonOL_WS, 
            gamma_mu_weight=gamma_mu_weight,
            gamma_cov_weight=gamma_cov_weight, 
            display_intermeds=show_pw_mnt_ptchs, verbose=verbose)


# Measure 12: Pairwise distributional patch matching - Symmetric KL divergence (Jeffrey's divergence)
# Measure 13: Pairwise distributional patch matching - Wasserstein-2 metric
# Measure 14: Pairwise distributional patch matching - Hellinger distance
# Measure 15: Pairwise distributional patch matching - Bhattacharyya distance
# Measure 16: Pairwise distributional patch matching - Mean-normed FM-eigendistance (FM-ATF)
# Based on the Forstner-Moonen metric on covariance matrices and the Bhattacharyya-normalized
#   mean-matching term, similar to Abou-Moustafa and Ferrie, 2012.
@timing_decorator()
def patchwise_moment_dist_c(img, mask, mode, pw_mnt_dist_nonOL_WS=[3,4], show_pw_mnt_ptchs=False, verbose=False, timing=False):
    return pairwise_moment_distances(img, mask, 
            block_cuts        = pw_mnt_dist_nonOL_WS, # TODO option for overlapping as well
            gamma_mu_weight   = None,
            gamma_cov_weight  = None,
            display_intermeds = show_pw_mnt_ptchs, 
            verbose           = verbose,
            mode              = mode )