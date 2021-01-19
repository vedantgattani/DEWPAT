""" Command line interface to visual_acuity.py
"""

import argparse, skimage, os
from module.visual_acuity import acuity_view


def main():
    ### >> Argument parsing << ###
    parser = argparse.ArgumentParser(
                description='Acuity View',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument('input', type=str, help='Input: either a folder or an image')
    parser.add_argument('width', type=float, help='The real width of the image.')
    parser.add_argument('distance', type=float,
                help='The viewing distance. Must be in the same units as the real width.')
    parser.add_argument('output', type=str, help='Specifies output directory to which the blurred images are saved.')
    parser.add_argument('--MRA', type=float, help='The minimum resolvable angle of the viewer.',
                default=0.08)
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                help='Whether to print verbosely while running')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        usables = [ '.jpg', '.png' ]
        usables = list(set( usables + [ b.upper() for b in usables ] + 
                                      [ b.lower() for b in usables ] ))
        _checker = lambda k: any( k.endswith(yy) for yy in usables )
        targets = [ os.path.join(args.input, f) 
                    for f in os.listdir(args.input) if _checker(f) ]
        for t in targets: 
            main_helper(t, args)
    elif os.path.isfile(args.input):
        main_helper(args.input, args)
    else:
        raise ValueError('Non-existent target input ' + args.input)


def main_helper(img_path, args):
    img_file_basename = os.path.basename(img_path)
    img_file_basename_extless, _ = os.path.splitext(img_file_basename)

    if args.verbose: print(f'Blurring {img_file_basename}')
    blurred_img = acuity_view(img_path, args.width, args.distance, R=args.MRA)

    # save the image
    if not os.path.isdir(args.output): os.makedirs(args.output)
    fname = os.path.join(args.output, img_file_basename_extless + '_blurred.png')
    skimage.io.imsave(fname=fname, arr=blurred_img)


if __name__ == '__main__':
    main()
