import os, sys, utils, argparse, sklearn, skimage, numpy as np, numpy.random as npr
import warnings, matplotlib.pyplot as plt
from skimage import data, segmentation, color
from skimage.future import graph
from sklearn import cluster

def main():
    ### Argument parsing ###
    parser = argparse.ArgumentParser(description='Image clustering/segmentation analysis.')
    parser.add_argument('input', type=str, help='Input: either a folder or an image')
    parser.add_argument('--labeller', dest='labeller', type=str, default='optics',
        choices=['affinity_prop', 'dbscan', 'optics', 'graph_cuts'],
        help='Labelling (segmentation and/or clustering) algorithm.')
    parser.add_argument('--blur', type=float, default=0.0,
        help='Specify Gaussian blur standard deviation applied to the image (default: none)')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
        help='Whether to print verbosely while running')
    # Labeller params
    group_g = parser.add_argument_group('Labeller parameters')
    group_g.add_argument('--dbscan_eps', dest='dbscan_eps', type=float, default=0.1,
        help='Epsilon neighbourhood used in DBSCAN.')
    group_g.add_argument('--dbscan_min_neb_size', dest='dbscan_min_neb_size', type=int, default=1,
        help='Min samples needed for core point designation in a neighbourhood. Used in DBSCAN.')
    group_g.add_argument('--gc_compactness', dest='gc_compactness', type=float, default=10.0,
        help='Graph cuts superpixel compactness.')
    group_g.add_argument('--gc_n_segments', dest='gc_n_segments', type=int, default=100,
        help='Graph cuts n_segments used in initialization.')
    group_g.add_argument('--gc_slic_sigma', dest='gc_slic_sigma', type=float, default=0.0,
        help='Graph cuts Gaussian kernel width for superpixel initialization.')
    
    # Analysis
    group_a = parser.add_argument_group('Analysis parameters')
    group_a.add_argument('--normalize_matrix', action='store_true',
        help='Whether to normalize the transition matrix')
    group_a.add_argument('--no_print_transitions', action='store_true',
        help='Pass flag to prevent printing the transition matrix')
    # Visualization
    parser.add_argument('--display', dest='display', action='store_true',
        help='Whether to display the resulting labelling')
    args = parser.parse_args()

    ### Read in the image ###
    img, R, G, B, mask = utils.load_helper(args.input) 
    H, W, C = img.shape
    if mask is None: mask = np.ones( (H,W) )
    if args.verbose: print('Loaded', args.input)
    # Gaussian filter if desired
    if args.blur > 1e-8:
        if args.verbose: print("\tBlurring with sigma =", args.blur)
        img = gauss_filter(img, args.blur)
    if args.verbose: 
        print('\tShape (%d,%d,%d)\n\tmin/max vals:' % (H,W,C), 
              img.min(0).min(0), img.max(0).max(0))
        print('\tNum masked values:', H*W - mask.sum(), "/", H*W)

    ### Label (cluster/segment) the image ###
    label_image = label(img, mask, args.labeller, args)
    
    ### Compute transition matrix ###
    M = transition_matrix(label_image, args.normalize_matrix, # TODO
            print_M = (not args.no_print_transitions) )

    ### Visualization ###
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
    if   method == 'affinity_prop': 
        clusterer = cluster.AffinityPropagation(affinity = 'euclidean',
                                                damping = 0.5) 
    elif method == 'dbscan':          
        clusterer = cluster.DBSCAN(eps = args.dbscan_eps, 
                                   min_samples = args.dbscan_min_neb_size,
                                   metric = 'euclidean') 
    elif method == 'optics':          
        clusterer = cluster.OPTICS(min_samples = 20,
                                   max_eps = np.inf,
                                   xi = 0.05)
    else:
        raise ValueError('Unknown method' + method)
    return clusterer.fit_predict(X)

def cluster_based_img_seg(image, mask, method, args):
    """
    Use spatially unaware vector-space clustering for label assignment.
    BG label: -1
    """
    # Extract colour-based vectors
    # TODO micropatches option
    vecs, indices = extract_vecs(image, mask) # TODO 
    # Cluster colour vectors to obtain labels
    vec_labels    = cluster_vecs(vecs) 
    # Post-process to remove noise 
    # E.g., noise labels are -1 in OPTICS and DBSCAN
    vec_labels    = post_process_labels(vec_labels, vec, indices, mask) # TODO
    # Use the image positions to reform the image
    # Background pixels are labeled -1
    label_image   = reform_label_image(vec_labels, indices, mask) # TODO
    return label_image

def reform_label_image(vec_labels, indices, mask):
    H, W = mask.shape
    # TODO 

    return label_image

def post_process_labels(vec_labels, vec, indices, mask):
    # TODO

    # Make sure no invalid values
    assert not any(vec_labels == -1), "Unexpected noise label. Try changing the settings."
    return vec_labels

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
#    for a in ax: a.axis('off')
    plt.tight_layout()

def transition_matrix(labels, normalize, print_M):
    """
    Input: label image (M x N, integer in [1,K])
    Output: K x K transition count/probability matrix (M_ij = count(i->j))
    """
    pass
    # TODO

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