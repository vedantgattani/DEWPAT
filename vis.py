"""
Implements several functions useful for visualization of images, particularly color distributions.
"""

import os, sys, numpy as np, skimage, argparse
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import imdisplay

def main(args):
    assert os.path.exists(args.input), "Input " + args.input + " does not exist."
    img = io.imread(args.input)
    H, W, C = img.shape
    print("Loaded image", args.input, "(Size: %s)" % str(img.shape))
    # Computing mask
    alpha_present = C == 4
    if alpha_present:
        print("\tAlpha channel present. Generating mask.")
        midvalue = 128
        orig_mask = img[:,:,3]
        bool_mask = (orig_mask > midvalue)
        int_mask = bool_mask.astype(int) 
        img = img[:,:,0:3]
        R, G, B = (img[:,:,0][bool_mask], img[:,:,1][bool_mask], img[:,:,2][bool_mask])
    else:
        print("\tNo alpha channel present.")
        R, G, B = img[:,:,0].flatten(), img[:,:,1].flatten(), img[:,:,2].flatten()
        orig_mask = None
    # At this point, img is always H X W x 3; mask is either None or H x W (boolean)

    #---------------------------------------------------------------------------------#

    ### Show the original image ###
    if args.show_img: display_orig(img, orig_mask, args)

    ### 1D RGB histograms ###
    if args.hist_rgb_1d: plot_1D_rgb(R, G, B)

    ### 3D RGB histograms ###
    if args.hist_rgb_3d: plot_3D_rgb(R, G, B)

    #> Final display activation <#
    plt.show()


def display_orig(img, orig_mask, args):
    imdisplay(img, title="Original Image (%s)" % args.input)
    if not orig_mask is None: 
        fig, ax = plt.subplots()
        ax.imshow(orig_mask) 
        plt.title('Mask (%s)' % args.input)

def plot_3D_rgb(R, G, B, nbins=8):
    print("Generating 3D RGB histogram")
    rngs= [(0, 255) for _ in range(3)]
    h, e = np.histogramdd([R, G, B], bins=nbins, range=rngs, density=True)
    from utils import histogram3dplot as histo3d
    fig = plt.figure(figsize=(8,8))
    histo3d(h, e, fig=fig)

def plot_1D_rgb(R,G,B,nbins=40):
    """
    Displays a plot with the 1D pixel value distributions per channel.
    """
    print("Generating 1D RGB histograms")
    cm_names = ("Reds", "Greens", "Blues")
    title_names = ["Histogram of %s Pixel Values" % s for s in ("Red", "Green", "Blue") ]
    # https://stackoverflow.com/questions/37360568/python-organisation-of-3-subplots-with-matplotlib
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(15,5))
    def gen_subplot(pos):
        V = R if pos == 0 else (G if pos == 1 else B)
        ax1 = fig.add_subplot(gs[0,pos])
        cm1 = plt.cm.get_cmap(cm_names[pos])
        n, bins, patches = ax1.hist(V, bins=nbins, density=1, range=(0,255), 
                                    linewidth=1.2, edgecolor='black') # color overridden
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers) # scale values to interval [0,1]
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm1(c))
        plt.title(title_names[pos])
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    gen_subplot(0) # Red
    gen_subplot(1) # Green
    gen_subplot(2) # Blue

if __name__ == '__main__':
    overall_desc = 'Implements several functions useful for visualization of images, particularly color distributions.'
    parser = argparse.ArgumentParser(description=overall_desc)
    # Input image
    parser.add_argument('input', type=str, help='Input: either a folder or an image')
    # Plotting options
    parser.add_argument('--show_img', dest='show_img', action='store_true',
        help='Whether to display the input image')
    parser.add_argument('--hist_rgb_1d', dest='hist_rgb_1d', action='store_true',
        help='Displays a set of 1D histograms of RGB pixel values')
    parser.add_argument('--hist_rgb_3d', dest='hist_rgb_3d', action='store_true',
        help='Displays a 3D histogram of RGB pixel values')
    args = parser.parse_args()
    main(args)