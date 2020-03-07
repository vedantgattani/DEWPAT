import os, sys, utils, argparse, sklearn, skimage, numpy as np, numpy.random as npr
import warnings, matplotlib.pyplot as plt
from skimage import data, segmentation, color
from skimage.future import graph
from sklearn import cluster
from scipy import interpolate
from collections import Counter

# TODO save json of counters and matrices for folders

def main():
    ### Argument parsing ###
    parser = argparse.ArgumentParser(description='Image clustering/segmentation analysis.')
    parser.add_argument('input', type=str, help='Input: either a folder or an image')
    parser.add_argument('--labeller', dest='labeller', type=str, default='kmeans',
        choices=['affinity_prop', 'dbscan', 'optics', 'graph_cuts', 'kmeans'],
        help='Labelling (segmentation and/or clustering) algorithm.')
    parser.add_argument('--blur', type=float, default=1.0,
        help='Specify Gaussian blur standard deviation applied to the image (default: 1)')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
        help='Whether to print verbosely while running')
    parser.add_argument('--resize', type=float, default=0.5,
        help='Specify scalar resizing. E.g., 0.5 halves the image size; 2 doubles it. (default: 0.5)')
    parser.add_argument('--write_mean_segs', action='store_true', dest='write_mean_segs',
                        help='Writes the segmented image(s) with cluster-mean values. Requires mean_seg_output_dir.')
    parser.add_argument('--mean_seg_output_dir', default=None,
                        help='Specifies the folder to which saved mean segments must be written')
    
    # Labeller params
    group_g = parser.add_argument_group('Labeller parameters')
    #--
    group_g.add_argument('--dbscan_eps', dest='dbscan_eps', type=float, default=1.0,
        help='Epsilon neighbourhood used in DBSCAN.')
    group_g.add_argument('--dbscan_min_neb_size', dest='dbscan_min_neb_size', type=int, default=20,
        help='Min samples needed for core point designation in a neighbourhood. Used in DBSCAN.')
    #--
    group_g.add_argument('--gc_compactness', dest='gc_compactness', type=float, default=10.0,
        help='Graph cuts superpixel compactness.')
    group_g.add_argument('--gc_n_segments', dest='gc_n_segments', type=int, default=300,
        help='Graph cuts n_segments used in initialization.')
    group_g.add_argument('--gc_slic_sigma', dest='gc_slic_sigma', type=float, default=0.0,
        help='Graph cuts Gaussian kernel width for superpixel initialization.')    
    #--
    group_g.add_argument('--kmeans_k', dest='kmeans_k', default=None,
        help='Number of clusters for K-means. (Overrides settings for automatic parameter determination).')    
    group_g.add_argument('--kmeans_auto_bounds', dest='kmeans_auto_bounds', type=str, default="2,6",
        help='Upper and lower bounds on the number of clusters searched over for kmeans. (e.g., 2,6)')    
    group_g.add_argument('--kmeans_auto_crit', dest='kmeans_auto_crit', type=str, default='davies_bouldin',
        choices=['silhouette', 'davies_bouldin', 'calinski_harabasz'],
        help='Choice of criterion for choosing the best k-value (either mean silhouette coefficient,' 
              'Davies-Bouldin score, or Calinski-Harabasz index).')    
    # Analysis
    group_a = parser.add_argument_group('Analysis parameters')
    group_a.add_argument('--normalize_matrix', action='store_true',
        help='Whether to normalize the transition matrix')
    group_a.add_argument('--no_print_transitions', action='store_true',
        help='Pass flag to prevent printing the transition matrix')
    group_a.add_argument('--keep_bg', action='store_true',
        help='Keeps the background label when computing the transition matrix')
    # Visualization
    # TODO heatmap with labels on scale (colorbar)
    parser.add_argument('--display', dest='display', action='store_true',
        help='Whether to display the resulting labelling')
    args = parser.parse_args()

    if args.write_mean_segs:
        assert not args.mean_seg_output_dir is None, "--write_mean_segs requires --mean_seg_output_dir"

    ### Handle images ###
    if os.path.isdir(args.input):
        usables = [ '.jpg', '.png' ]
        usables = list(set( usables + [ b.upper() for b in usables ] + 
                                      [ b.lower() for b in usables ] ))
        _checker = lambda k: any( k.endswith(yy) for yy in usables )
        targets = [ os.path.join(args.input, f) 
                    for f in os.listdir(args.input) if _checker(f) ]
        for t in targets: main_helper(t, args)
    elif os.path.isfile(args.input):
        main_helper(args.input, args)
    else:
        raise ValueError('Non-existent target input ' + args.input)
    
def main_helper(img_path, args):
    #img_filename_extless, img_file_extension = os.path.splitext(img_path)
    img_file_basename = os.path.basename(img_path)
    img_file_basename_extless = os.path.splitext(img_file_basename)[0]

    img = skimage.io.imread(img_path)
    if args.verbose: print('Loaded', img_path)
    n_channels = img.shape[2]

    # Downscale image, if desired
    resize_factor_main = args.resize
    assert resize_factor_main > 0.0, "--resize must be positive"
    running_resize = False
    if abs(resize_factor_main - 1.0) > 1e-4:
        running_resize = True
        if args.verbose: print("Orig shape", img.shape, "| Resizing by", resize_factor_main)
        img = skimage.transform.rescale(img, scale=resize_factor_main,
                anti_aliasing=True, multichannel=True)
        img = utils.conv_to_ubyte(img)
        if args.verbose: print("Resized dims:", img.shape)
    mask = None
    if n_channels == 4:
        a = img[:, :, 3]
        img = img[:, :, :3]
        mask = a.copy()
        mask[ a >  128 ] = 1 # 255
        mask[ a <= 128 ] = 0
        # Zero out the background
        img *= mask[:,:,np.newaxis]
    
    H, W, C = img.shape
    if mask is None: mask = np.ones( (H,W) )
    # Gaussian filter if desired
    if args.blur > 1e-8:
        if args.verbose: print("\tBlurring with sigma =", args.blur)
        img = gauss_filter(img, args.blur)
    if args.verbose: 
        print('\tShape (%d,%d,%d)\n\tmin/max vals:' % (H,W,C), 
              img.min(0).min(0), img.max(0).max(0))
        print('\tNum masked values:', H*W - (mask > 0).sum(), "/", H*W)
    # Move from byte to float
    #img = skimage.img_as_float(img) # 255.0

    ### Label (cluster/segment) the image ###
    label_image = label(img, mask, args.labeller, args)
    # Write mean-cluster-valued image out if desired
    if args.write_mean_segs:
        int_mask = (mask*255).reshape(H,W,1)
        mean_segs = color.label2rgb(label_image, image = img, bg_label = -1, 
                                    bg_color = (0.0, 0.0, 0.0), kind = 'avg')
        mean_segs = np.concatenate( (mean_segs, int_mask), axis = 2 )
        if not os.path.isdir(args.mean_seg_output_dir):
            os.makedirs(args.mean_seg_output_dir)
        cfname = os.path.join(args.mean_seg_output_dir, 
                              img_file_basename_extless + ".mean_seg.png")
        if args.verbose: print('\tSaving mean seg image to', cfname)
        skimage.io.imsave(fname = cfname, arr = mean_segs)

    ### Compute transition matrix ###
    M = transition_matrix(label_image, args.normalize_matrix, 
            print_M = (not args.no_print_transitions), args=args )

    ### Visualization ###
    if args.display:
        vis_label_img(img, label_image, args)
    plt.show()

def label(image, mask, method, args):
    """
    Input: I (M x N x C)
    Output: Pixelwise labels (M x N), integer
    """
    if method == 'graph_cuts': return segment(image, method, mask, args)
    else:                      return cluster_based_img_seg(image, mask, method, args)

def cluster_vecs(X, method, args):
    """
    Input: X (S x D, row vector per datum)
    Output: cluster labels vector (S, integer)
    """
    X = X.astype(float)
    #print(X, type(X))
    if   method == 'affinity_prop': 
        clusterer = cluster.AffinityPropagation(affinity = 'euclidean',
                                                damping = 0.5) 
    elif method == 'dbscan':          
        clusterer = cluster.DBSCAN(eps = args.dbscan_eps, 
                                   min_samples = args.dbscan_min_neb_size,
                                   metric = 'euclidean') 
    elif method == 'optics':          
        clusterer = cluster.OPTICS(min_samples = 0.05,
                                   max_eps = np.inf,
                                   xi = 0.05)
    elif method == 'kmeans':
        kmeans_B = 100
        specified_k = args.kmeans_k
        if not (specified_k is None):
            if args.verbose: print("\tPreparing k-means clusterer with fixed k =", specified_k)
            clusterer = cluster.MiniBatchKMeans(n_clusters = int(args.kmeans_k), batch_size = kmeans_B)
        else:
            _min_k, _max_k = list(map(int, args.kmeans_auto_bounds.strip().split(","))) # min,max
            verb_str_a = "\tAttempting to automatically determine k. Searching over k ="
            if args.verbose: print(verb_str_a, _min_k, "to", _max_k)
            def run_clustering(nc_k):
                if args.verbose: print("\tOn k =", nc_k)
                return cluster.MiniBatchKMeans(n_clusters = nc_k, batch_size = kmeans_B)
            clusterers = [ (i, run_clustering(i)) 
                            for i in range(_min_k, _max_k+1) ]
            clustering_labels = [ (c[0], c[1].fit_predict(X)) for c in clusterers ]
            if args.kmeans_auto_crit == 'silhouette': 
                scorer_function = sklearn.metrics.silhouette_score
                _coef = 1.0
            elif args.kmeans_auto_crit == 'davies_bouldin': 
                scorer_function = sklearn.metrics.davies_bouldin_score
                _coef = -1.0 # Since lower is better
            elif args.kmeans_auto_crit == 'calinski_harabasz':
                scorer_function = sklearn.metrics.calinski_harabasz_score
                _coef = 1.0
            else:
                raise ValueError('Unknown scorer ' + args.kmeans_auto_crit)
            crit_scores = [ (i, scorer_function(X, labels)) for (i, labels) in clustering_labels ]
            best_ind = np.argmax([ _coef * a[1] for a in crit_scores ])
            if args.verbose:
                print("\tObtained the following scores with criterion %s:" % args.kmeans_auto_crit)
                for i, s in crit_scores:
                    print("\t\tk = %d -> %.3f" % (i,s))
                print('\tBest k:', crit_scores[best_ind][0])
            return clustering_labels[best_ind][1]
    else:
        raise ValueError('Unknown method' + method)
    if args.verbose: print("\tRunning clustering via", method)
    return clusterer.fit_predict(X)

def cluster_based_img_seg(image, mask, method, args):
    """
    Use spatially unaware vector-space clustering for label assignment.
    BG label: -1
    """
    # Extract colour-based vectors
    # TODO micropatches option
    vecs        = extract_vecs(image, mask) 
    if args.verbose: print("\tObtained vectors:", vecs.shape)
    # Cluster colour vectors to obtain labels
    vec_labels  = cluster_vecs(vecs, method, args) 
    if args.verbose: print("\t\tComplete")
    # Use the image positions to reform the image
    # Background pixels are labeled -1
    label_image = reform_label_image(vec_labels, mask) 
    if args.verbose: 
        print("\tFinished reforming label image")
        #labelset = set()
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

def vis_label_img(image, labels_img, args):
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
    #fig = plt.figure(figsize=(15, 5))
    fig, ax = plt.subplots(nrows=2, ncols=2) #, sharex=True, sharey=True) #, figsize=(25, 8))
    _q = [(0,0), (0,1), (1,0), (1,1)]
    _t = ['Orig Image', 'Segmentation', 'Seg Overlay', 'Mean Seg']
    for i, I in enumerate([image, pure_segs, over_segs, mean_segs]):
        #fig.add_subplot(1, i+1, i+1)
        j, k = _q[i]
        ax[j, k].imshow(I)
        ax[j, k].axis('off')
        ax[j, k].set_title(_t[i])
    # for a in ax: a.axis('off')
    plt.tight_layout()

def transition_matrix(L, normalize, print_M, args):
    """
    Input: label image (M x N, integer in [1,K])
    Output: K x K transition count/probability matrix (M_ij = count(i->j))
    """
    H, W = L.shape
    Hpp, Wpp = H + 1, W + 1
    allowed_labels = list(set(L.flatten().astype(int).tolist()))
    n_labels = len(allowed_labels)
    n_masked = (L == -1).sum()
    #print('p',allowed_labels)
    # Shifted label images
    L_left  = np.c_[ L, -np.ones(H) ]
    L_right = np.c_[ -np.ones(H), L ]
    L_up    = np.r_[ [-np.ones(W)], L ]
    L_down  = np.r_[ L, [-np.ones(W)] ]
    #print(H,W,L_up.shape,L_down.shape, L_left.shape, L_right.shape)
    horz_shift = np.stack( (L_left, L_right), axis=-1 ).reshape(H*Wpp, 2)
    vert_shift = np.stack( (L_up, L_down),    axis=-1 ).reshape(Hpp*W, 2)
    all_tuples = np.concatenate( (horz_shift, vert_shift), axis = 0 ).astype(int).tolist()
    if args.verbose: print('\tAssembled', len(all_tuples), 'edge pairs for transition estimation')
    if not args.keep_bg:
        if -1 in allowed_labels: allowed_labels.remove(-1) # 
        n_labels = len(allowed_labels)
        _ATN = len(all_tuples)
        all_tuples = [ elem for elem in all_tuples if not (-1 in elem) ]
        if args.verbose: 
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
    if print_M or args.verbose: 
        if args.verbose: print("Transition counts:", transition_counters)
        print('Labels:', allowed_labels)
        if args.verbose: 
            print('Column sums:', M.sum(1), '| Total:', np.tril(M,k=0).sum())
            extot = 2*H*W + W + H
            print('N_expected_transitions:', extot, '(unmasked: %d)' % (extot - n_masked - H - W))
        print('Transition matrix')
        print(M)
    return M, transition_counters

def segment(image, method, mask, args):
    """ Spatially aware image segmentation """
    #if not (mask is None):
    #    H, W, C = image.shape
    #    image = np.concatenate( (image, mask.reshape(H,W,1)), axis=2)
    image = send_mask_to_minus_one(image, mask)
    if method == 'graph_cuts':
        labels1 = segmentation.slic(image, compactness=args.gc_compactness,
                                    n_segments=args.gc_n_segments, 
                                    sigma=args.gc_slic_sigma) #, compactness=gc_compactness)
        out1    = color.label2rgb(labels1, image, kind='avg')
        g       = graph.rag_mean_color(image, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)
        labels2 = send_mask_to_minus_one(labels2, mask)
        return labels2

def gauss_filter(img, blur_sigma):
    assert blur_sigma >= 0.0, "Untenable blur kernel width"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_ubyte( utils.gaussian_blur(img, blur_sigma) )
    return img

def send_mask_to_minus_one(image, mask):
    #print(mask[0:100,0:100], image[0:100,0:100])
    image[ mask <= 0 ] = -1 
    return image

#-------------------------#
if __name__ == '__main__':
    main()
#-------------------------#


#
