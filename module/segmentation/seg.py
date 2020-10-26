import argparse, sklearn, skimage, numpy as np, numpy.random as npr
import warnings, matplotlib.pyplot as plt, csv
from skimage import data, segmentation, color
from skimage.color import rgb2hsv
from skimage.future import graph
from sklearn import cluster
from scipy import interpolate
from collections import Counter
from ..utils import Formatter, gaussian_blur


### Label (cluster/segment) the image ###
def reorder_label_img(LI, verbose=False):
    if verbose: print('\tReordering label image')
    LI = np.rint(LI).astype(int)
    labels_list = LI.flatten().astype(int).tolist()
    c = Counter(labels_list)
    if verbose: print('\tOrig Counter:', c)
    ordered_targets = c.most_common() # [(int_key, count),...]
    curr_new_label = 1
    LI_new = np.copy(LI)
    for j, (int_key, count) in enumerate(ordered_targets):
        if int_key == -1: continue # Leave the background alone
        LI_new[ LI == int_key ] = curr_new_label
        curr_new_label += 1
    if verbose: print('\tNew Counter:', Counter(LI_new.flatten().astype(int).tolist()))
    return LI_new

def k_dependent_smaller_clusters_merging_threshold(k, kdep_param=0.05):
    # Start at some value for k=2, and anneal down as k increases
    if k <= 1: return 0.0 # No merging if k == 1
    T0 = kdep_param
    T  = T0 / (k - 1)
    return T
def merge_smaller_clusters(img, mask, LI, merge_small_clusters_method=None, small_cluster_merging_kdep_param=0.05,
                           fixed_cluster_size_merging_threshold=0.05, small_cluster_merging_dynamic_k=False, verbose=False):
    LI = np.rint(LI).astype(int)
    labels_list = LI.flatten().tolist()
    c = Counter(labels_list)
    labels_initial = list(c.keys())
    if verbose: print('Calling cluster merging:', merge_small_clusters_method)
    if merge_small_clusters_method == 'none': return LI
    elif merge_small_clusters_method == 'k_dependent':
        k_initial = len(labels_initial)
        if -1 in labels_initial: k_initial -= 1
        thresh = k_dependent_smaller_clusters_merging_threshold(k_initial, kdep_param=small_cluster_merging_kdep_param)
    elif merge_small_clusters_method == 'fixed':
        thresh = fixed_cluster_size_merging_threshold
    else: raise ValueError('Unknown merging method ' + str(merge_small_clusters_method))
    assert thresh >= 0.0 and thresh <= 1.0, "Untenable threshold value " + str(thresh)
    if verbose: print('Attempting merge. Chose merging threshold', thresh)
    # Now have the threshold, need to merge the smaller clusters below that threshold
    def attempt_merge(currLI):
        """ Runs a single merge iteration. Returns None if nothing was merged. """
        print('\tEntering single merge attempt')
        D = label_img_to_stats(img, mask, currLI)
        cluster_stats = D['cluster_info']['cluster_stats']
        current_labels = D['cluster_info']['allowed_labels']
        found_failure = False
        percents = [ (cluster_stats[clabel]['percent_member_pixels'], clabel)
                        for clabel in current_labels if not clabel == -1 ]
        _current_k = len(percents)
        sorted_percents = sorted(percents, key=lambda x: x[0]) # min percent at front
        if verbose: print('\t\tObtained percents', sorted_percents)
        curr_percent, label = sorted_percents[0] # We always merge the smallest one
        # If specified, we should recompute the threshold
        nonlocal thresh
        if small_cluster_merging_dynamic_k and merge_small_clusters_method == 'k_dependent':
            thresh = k_dependent_smaller_clusters_merging_threshold(_current_k)
            if verbose: print('\tRecomputed k-dep threshold:', thresh)
        if curr_percent < thresh:
            if verbose: print('\t\tDetected cluster size violation. Merging.')
            found_failure = True
            # Target colour (to destroy)
            targeted_mean_colour = cluster_stats[label]['mean_colour']
            # Distances in colour space to target
            dists = [ (( (  targeted_mean_colour 
                            - cluster_stats[olabel]['mean_colour']
                            )**2 ).sum(), olabel)
                        for olabel in current_labels 
                        if not (olabel in [-1, label]) ]
            sorted_oths = sorted(dists, key=lambda x: x[0], reverse=False) # min dist at front
            # Colour cluster label to replace the target with
            merge_into_target = sorted_oths[0][1] # <- cluster label
            if verbose:
                mean_oth = cluster_stats[merge_into_target]['mean_colour']
                oth_percent = cluster_stats[merge_into_target]['percent_member_pixels']
                print('\t\tMerging cluster %d (%s,%.3f) into %d (%s,%.3f)' %
                        (  label, str(targeted_mean_colour), curr_percent,
                            merge_into_target, str(mean_oth), oth_percent   ) )
            # We've found the closest cluster. Now to perform the merging.
            LI_copy = np.copy(currLI)
            LI_copy[ currLI == label ] = merge_into_target
        if found_failure: return LI_copy
        else: return None # Finished all merges
        #</ End of iterated merging routine />#
    # Iteratively attempt merging the smallest cluster until it no longer occurs 
    all_clear = False
    currLI = LI
    while not all_clear:
        merged_output = attempt_merge(currLI)
        if merged_output is None: # we are done
            all_clear = True
        else: # Try again since a merging occurred
            currLI = merged_output
    if verbose: print('\tFinished merging attempts')
    return currLI

def label_img_to_stats(img, mask, label_img, mean_seg_img=None, verbose=False):
    """
    Returns a dictionary of label img and initial image stats
    This includes a subdictionary of information per label set (cluster)
    """
    if mean_seg_img is None:
        mean_seg_img = color.label2rgb(label_img, image = img, bg_label = -1, 
                                       bg_color = (0.0, 0.0, 0.0), kind = 'avg')
    # 
    H, W, C = img.shape
    allowed_labels = list(set(label_img.flatten().astype(int).tolist()))
    n_labels = len(allowed_labels)
    n_masked = (label_img == -1).sum()
    n_unmasked = (H*W) - n_masked
    cluster_info = { 'n_labels' : n_labels, 'allowed_labels' : allowed_labels }
    cluster_stats = {}
    for label in allowed_labels:
        if label == -1: continue # Ignore bg
        bool_indexer = np.logical_and(mask != 0, label_img == label)
        orig_vals = img[bool_indexer] # Valid values in the current cluster
        mean_vals = mean_seg_img[bool_indexer]
        mean_cluster_value = orig_vals.mean(0)
        premeaned_val = mean_vals.mean(0)
        premeaned_val_hsv = rgb2hsv(premeaned_val.reshape(1,1,3) / 255.0)[0,0,:]
        cluster_stats[label] = {
            'label_number'          : label,
            'mean_C1'               : mean_cluster_value[0], # R (floats)
            'mean_C2'               : mean_cluster_value[1], # G
            'mean_C3'               : mean_cluster_value[2], # B
            'mean_C1_hsv'           : premeaned_val_hsv[0],  # H in [0,1], multiply by 360 to get degrees
            'mean_C2_hsv'           : premeaned_val_hsv[1],  # S
            'mean_C3_hsv'           : premeaned_val_hsv[2],  # V
            'n_member_pixels'       : orig_vals.shape[0],
            'percent_member_pixels' : orig_vals.shape[0] / n_unmasked,
            'mean_colour'           : premeaned_val, # Integer array
        }
    cluster_info['cluster_stats'] = cluster_stats
    D = { 'H'                 : H,
          'W'                 : W,
          'total_pixels'      : H*W,
          'n_masked_pixels'   : n_masked,
          'n_unmasked_pixels' : n_unmasked,
          'cluster_info'      : cluster_info,
    }
    if verbose: 
        print('Gathered seg image info:')
        _glob_form = Formatter()
        _glob_form.print_dict(D)
    return D

def label(image, mask, method, clustering_colour_space='rgb', dbscan_eps=1.0, dbscan_min_neb_size=20, kmeans_k=None,
          kmeans_auto_bounds='2,6', kmeans_auto_crit='davies_bouldin', gc_compactness=10.0, gc_n_segments=300,
          gc_slic_sigma=0.0,verbose=False):
    """
    Input: I (M x N x C)
    Output: Pixelwise labels (M x N), integer
    """
    # Convert to desired colour space
    if not clustering_colour_space == 'rgb':
        if verbose: 
            _K = clustering_colour_space 
            _app = { 'cie' : '(CIE-1931-RGB)', 'lab' : '(CIE-LAB)', 'hsv' : '' }
            print('Converting to colour space', _K, _app[_K]) 
        if clustering_colour_space == 'hsv':
            image = np.rint( rgb2hsv(image/255.0) * 255.0 ).astype(int) 
        if clustering_colour_space == 'cie': # CIE1931
            from skimage.color import rgb2rgbcie
            image = np.rint( rgb2rgbcie(image/255.0) * 255.0 ).astype(int)
        if clustering_colour_space == 'lab': # CIE-LAB / CIE L*a*b*
            # LAB may be the most perceptually principled choice since (in humans)
            # it can be used to compute more perceptual colour differences.
            from skimage.color import rgb2lab # Bounds: [0,100], [-128,128], [-128,128]
            image = np.rint( rgb2lab(image/255.0) ).astype(int) 
    # Perform clustering
    if method == 'graph_cuts': return segment(image, method, mask, gc_compactness=gc_compactness, gc_n_segments=gc_n_segments,
                                              gc_slic_sigma=gc_slic_sigma)
    else:                      return cluster_based_img_seg(image, mask, method, dbscan_eps=dbscan_eps,
                                                            dbscan_min_neb_size=dbscan_min_neb_size, kmeans_k=kmeans_k,
                                                            kmeans_auto_bounds=kmeans_auto_bounds,
                                                            kmeans_auto_crit=kmeans_auto_crit, verbose=verbose)

def cluster_vecs(X, method, dbscan_eps=1.0, dbscan_min_neb_size=20, kmeans_k=None, kmeans_auto_bounds='2,6',
                 verbose=False, kmeans_auto_crit='davies_bouldin'):
    """
    Input: X (S x D, row vector per datum)
    Output: cluster labels vector (S, integer)
    """
    X = X.astype(float)
    if   method == 'affinity_prop': 
        clusterer = cluster.AffinityPropagation(affinity = 'euclidean',
                                                damping = 0.5) 
    elif method == 'dbscan':          
        clusterer = cluster.DBSCAN(eps = dbscan_eps, 
                                   min_samples = dbscan_min_neb_size,
                                   metric = 'euclidean') 
    elif method == 'optics':          
        clusterer = cluster.OPTICS(min_samples = 0.05,
                                   max_eps = np.inf,
                                   xi = 0.05)
    elif method == 'kmeans':
        kmeans_B = 100
        specified_k = kmeans_k
        if not (specified_k is None):
            if verbose: print("\tPreparing k-means clusterer with fixed k =", specified_k)
            clusterer = cluster.MiniBatchKMeans(n_clusters = int(kmeans_k), batch_size = kmeans_B)
        else:
            _min_k, _max_k = list(map(int, kmeans_auto_bounds.strip().split(","))) # min,max
            verb_str_a = "\tAttempting to automatically determine k. Searching over k ="
            if verbose: print(verb_str_a, _min_k, "to", _max_k)
            def run_clustering(nc_k):
                if verbose: print("\tOn k =", nc_k)
                return cluster.MiniBatchKMeans(n_clusters = nc_k, batch_size = kmeans_B)
            clusterers = [ (i, run_clustering(i)) 
                            for i in range(_min_k, _max_k+1) ]
            clustering_labels = [ (c[0], c[1].fit_predict(X)) for c in clusterers ]
            if kmeans_auto_crit == 'silhouette': 
                scorer_function = sklearn.metrics.silhouette_score
                _coef = 1.0
            elif kmeans_auto_crit == 'davies_bouldin': 
                scorer_function = sklearn.metrics.davies_bouldin_score
                _coef = -1.0 # Since lower is better
            elif kmeans_auto_crit == 'calinski_harabasz':
                scorer_function = sklearn.metrics.calinski_harabasz_score
                _coef = 1.0
            else:
                raise ValueError('Unknown scorer ' + kmeans_auto_crit)
            crit_scores = [ (i, scorer_function(X, labels)) for (i, labels) in clustering_labels ]
            best_ind = np.argmax([ _coef * a[1] for a in crit_scores ])
            if verbose:
                print("\tObtained the following scores with criterion %s:" % kmeans_auto_crit)
                for i, s in crit_scores:
                    print("\t\tk = %d -> %.3f" % (i,s))
                print('\tBest k:', crit_scores[best_ind][0])
            return clustering_labels[best_ind][1]
    else:
        raise ValueError('Unknown method' + method)
    if verbose: print("\tRunning clustering via", method)
    return clusterer.fit_predict(X)

def cluster_based_img_seg(image, mask, method, dbscan_eps=1.0, dbscan_min_neb_size=20, kmeans_k=None,
                          kmeans_auto_bounds='2,6', kmeans_auto_crit='davies_bouldin', verbose=False):
    """
    Use spatially unaware vector-space clustering for label assignment.
    BG label: -1
    """
    # Extract colour-based vectors
    # TODO micropatches option
    vecs        = extract_vecs(image, mask) 
    if verbose: print("\tObtained vectors:", vecs.shape)
    # Cluster colour vectors to obtain labels
    vec_labels  = cluster_vecs(vecs, method, dbscan_eps=dbscan_eps, dbscan_min_neb_size=dbscan_min_neb_size, kmeans_k=kmeans_k,
                               kmeans_auto_bounds=kmeans_auto_bounds, kmeans_auto_crit=kmeans_auto_crit, verbose=verbose) 
    if verbose: print("\t\tComplete")
    # Use the image positions to reform the image
    # Background pixels are labeled -1
    label_image = reform_label_image(vec_labels, mask) 
    if verbose: print("\tFinished reforming label image")
    return label_image

def extract_vecs(image, mask): return image[ mask > 0 ]

def reform_label_image(vec_labels, mask):
    H, W = mask.shape
    # Form a label image (initialize to -2)
    label_image = np.zeros( (H,W) ) - 2
    # Set unmasked parts
    label_image[mask > 0] = vec_labels
    ### Post-process to remove noise labels ###
    # E.g., noise labels are -1 in OPTICS and DBSCAN -> convert to nan
    label_image[ label_image == -1 ] = np.nan
    # Convert to masked array
    a = np.ma.masked_invalid(label_image)
    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, H))
    # Retrieve only valid values
    x1 = xx[ ~a.mask ] # x indices
    y1 = yy[ ~a.mask ] # y indices
    new_data = a[ ~a.mask ] # unmasked data values
    # Interpolate via nearest neighbours
    G = interpolate.griddata( (x1, y1), new_data.ravel(),
                              (xx, yy), method = 'nearest' )
    label_image = G
    ### Move -2 -> -1 (background signal) ###
    label_image[ label_image == -2 ] = -1
    assert not np.isnan(label_image).any() 
    return label_image

def vis_label_img(image, labels_img):
    """
    Visualization method for clustered/segmented image.
    Display segmentation, overlaid segmentation, and mean-value segmentation.
    """
    # Segmentation
    pure_segs = color.label2rgb(labels_img)
    # Overlaid segmentation
    over_segs = color.label2rgb(labels_img, image=image, 
                                alpha=0.25, bg_label=-1, 
                                bg_color=(0,0,0), image_alpha=1, 
                                kind='overlay')
    # Mean value segmentation
    mean_segs = color.label2rgb(labels_img, image=image, 
                                alpha=0.25, bg_label=-1, 
                                bg_color=(0,0,0), image_alpha=1, 
                                kind='avg')
    # Show image triplet
    fig, ax = plt.subplots(nrows=2, ncols=2) #, sharex=True, sharey=True) #, figsize=(25, 8))
    _q = [(0,0), (0,1), (1,0), (1,1)]
    _t = ['Orig Image', 'Segmentation', 'Seg Overlay', 'Mean Seg']
    for i, I in enumerate([image, pure_segs, over_segs, mean_segs]):
        j, k = _q[i]
        ax[j, k].imshow(I)
        ax[j, k].axis('off')
        ax[j, k].set_title(_t[i])
    plt.tight_layout()

def transition_matrix(L, normalize, print_M, keep_bg=False, verbose=False):
    """
    Input: label image (M x N, integer in [1,K])
    Output: K x K transition count/probability matrix (M_ij = count(i->j))
    """
    H, W = L.shape
    Hpp, Wpp = H + 1, W + 1
    allowed_labels = list(set(L.flatten().astype(int).tolist()))
    n_labels = len(allowed_labels)
    n_masked = (L == -1).sum()
    # Shifted label images
    L_left  = np.c_[ L, -np.ones(H) ]
    L_right = np.c_[ -np.ones(H), L ]
    L_up    = np.r_[ [-np.ones(W)], L ]
    L_down  = np.r_[ L, [-np.ones(W)] ]
    horz_shift = np.stack( (L_left, L_right), axis=-1 ).reshape(H*Wpp, 2)
    vert_shift = np.stack( (L_up, L_down),    axis=-1 ).reshape(Hpp*W, 2)
    all_tuples = np.concatenate( (horz_shift, vert_shift), axis = 0 ).astype(int).tolist()
    if verbose: print('\tAssembled', len(all_tuples), 'edge pairs for transition estimation')
    if not keep_bg:
        if -1 in allowed_labels: allowed_labels.remove(-1) # 
        n_labels = len(allowed_labels)
        _ATN = len(all_tuples)
        all_tuples = [ elem for elem in all_tuples if not (-1 in elem) ]
        if verbose: 
            print("\t\tPost-bg removal:", len(all_tuples), "remaining (%d removed)" % 
                    (_ATN - len(all_tuples)))
            print("\t\tExpected", 2*(H+W), 'missing values due to appended boundaries')
    # Stringify the tuples
    all_tuples = map(lambda s: ",".join(map(str, s)), all_tuples)
    transition_counters = Counter(all_tuples)
    # Generate matrix form of transition
    M = np.zeros( (n_labels, n_labels) )
    for i, labi in enumerate(allowed_labels):
        for j, labj in enumerate(allowed_labels):
            key1 = str(labi) + "," + str(labj)
            key2 = str(labj) + "," + str(labi)
            M[i,j] = int(transition_counters[key1] + transition_counters[key2])
    if normalize: M = M / M.sum(axis=1)[:,np.newaxis]
    else:         M = M.astype(int)
    if print_M or verbose: 
        if verbose: print("Transition counts:", transition_counters)
        print('Labels:', allowed_labels)
        if verbose: 
            print('Column sums:', M.sum(1), '| Total:', np.tril(M,k=0).sum())
            extot = 2*H*W + W + H
            print('N_expected_transitions:', extot, '(unmasked: %d)' % (extot - n_masked - H - W))
        print('Transition matrix')
        print(M)
    return M, transition_counters

def segment(image, method, mask, gc_compactness=10.0, gc_n_segments=300, gc_slic_sigma=0.0):
    """ Spatially aware image segmentation """
    #if not (mask is None):
    #    H, W, C = image.shape
    #    image = np.concatenate( (image, mask.reshape(H,W,1)), axis=2)
    image = send_mask_to_minus_one(image, mask)
    if method == 'graph_cuts':
        labels1 = segmentation.slic(image, compactness=gc_compactness,
                                    n_segments=gc_n_segments, 
                                    sigma=gc_slic_sigma) #, compactness=gc_compactness)
        out1    = color.label2rgb(labels1, image, kind='avg')
        g       = graph.rag_mean_color(image, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)
        labels2 = send_mask_to_minus_one(labels2, mask)
        return labels2

def gauss_filter(img, blur_sigma):
    assert blur_sigma >= 0.0, "Untenable blur kernel width"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_ubyte( gaussian_blur(img, blur_sigma) )
    return img

def send_mask_to_minus_one(image, mask):
    image[ mask <= 0 ] = -1 
    return image
