# General overview statement

DEWPAT currently exists as a series of Python scripts divided between two branches. Note that both branches should technically contain all of the scripts, but for best and most up to date functionality you should use the script from the branch designated below.

### master branch

- img_complexity.py: main script that includes functionality to compute and visualize complexity across both sRGB and multispectral images. Note that if using mspec images, need to run --mspec along with other arguments

- vis.py: includes visualization options (including 1D colour histogram) for sRGB images only


### dev branch

- seg.py: includes functionality to segment, visualize, and output colour statistics for sRGB images only

- preprocess.py: includes functionality to blur both sRGB and multispectral images to model visual acuity and viewing distance using AcuityView (Caves & Johnsen, 2017).




# Running DEWPAT manuscript examples


### DEWPAT Section 3.1. Worked Examples

### Example 1
#### Use img_complexity.py in master branch

sRGB images of beetle elytra are in folder beetle_elytra. use the following code to calculate 4 measures of complexity on the images and save the results in beetle_complexity.csv

python img_complexity.py beetle_elytra --grad_mag --diff_shannon_entropy_patches --global_patch_covar --pwg_bhattacharyya_div >beetle_complexity.csv

### Example 2
#### Use seg.py in dev branch

sRGB images of anole dewlaps are in folder anole_dewlaps. use the following code to run DEWPAT’s auto segmentation feature to segment images into k-clusters, coloured by median colour (k-median clustering) and put the resulting images in the folder anole_segs and exported an output file that, for each image, reports the colour value of each detected colour cluster in three different colour spaces (RGB, HSV, and CIELAB) as well as the frequency and percentage of pixels in each cluster in anole_colour_stats.csv 

python seg.py --no_print_transitions anole_dewlaps --labeller kmeans  --write_median_segs --median_seg_output_dir anole_segs --seg_median_stats_output_file anole_colour_stats.csv 


### Example 3
#### Use img_complexity.py in master branch. remember to pass --mspec for mspec images

sRGB images of flower petals are in folder flower_ex_vis_flowers and mspec images of flower petals are in folder flower_ex_bee_flower. use the following code to calculate global patch covariance on the images and save the results.

python img_complexity.py flower_ex_vis_flowers --global_patch_covar >flower_vis_complexity.csv

python img_complexity.py --mspec flower_ex_bee_flowers --global_patch_covar >flower_bee_complexity.csv


### DEWPAT figures
We've also included the images and code needed to create the plots in some of the figures to illustrate further functionality

### figure 2
#### Can use img_complexity.py in either branch but we recommend master branch

sRGB images of anole dewlap cross sections are in folder fig2. use the following code to view visual output from DEWPAT representing different dimensions of pattern complexity 

python img_complexity.py fig2 --show_all


### figure 3
#### Use vis.py in master branch

NOTE: if you want to change the bin numbers for the histogram, find the nbins=75 in the vis.py script and change the number to the desired bin number.

sRGB images of anole dewlap cross sections are loose in folder and in folder fig3. use the following code to view various visual output plots from DEWPAT 
python vis.py porcatus.png --all
python vis.py gemmosus.png --all

To output .csv file with pixel frequencies in each colour bin run:
python vis.py fig3 --write_1d_histo_vals --output_file fig3.csv