# FOVEA: Preoperative and Intraoperative Retinal Fundus Images with Optic Disc and Retinal Vessel Annotations

This repository contains code used in the processing and analysis of the FOVEA dataset in the file ``fovea_utilities.py``. The necessary python packages can be installed with:

``pip install -r requirements.txt``

The following sections group the available functions by topic and explain their usage. For details on how to run them, see the docstrings and type hints provided in the code.


## Processing

The main function to create binary masks from the raw annotations is ``create_masks()``. If pointed at the main FOVEA folder, it will load the raw annotations and create binary masks using ``mask_from_raw()``. These are then combined with annotation skeletons created with ``skel_from_raw()``, and ``find_contiguous_features()`` followed by ``prune_features_by_size()`` is run on the result to cut out accidental pencil touches during annotation or otherwise unconnected features below the given threshold.

The resulting binary masks - one per annotator, modality, and domain, i.e. eight per id number - are saved in the given output folder. By default, the original images are copied into the new folder as well, though that can be controlled by the ``copy_img`` boolean flag.

Optionally, an additional folder called "control" will be created inside the target folder, with:
* A json file called ``feat_sizes.json`` with data on the sizes of contiguous areas in the mask before and after combining the first binary mask with the skeleton
* Control images showing retained annotations in green and pruned areas in red, citing the number of cut pixels in the file name. This uses the ``control_img_from_masks()`` and ``blend_images()`` functions.


## Analysis

Statistics about the dataset or more specifically the binary masks can be obtained with ``analyse_dataset()``. It will read the masks at the given path and calculate coverage (percentage of pixels that are annotated as optic disc or retinal vessels, respectively) as well as a number of mask agreement statistics, including the DICE score. This uses both ``calc_confusion_matrix()`` and ``calc_stats()``. The result is saved as ``stats.json`` in a new "stats" folder in the given path, along with images showing annotator agreement for optic disc and retinal vessel annotations.

The ``feat_sizes.json`` file optionally created by the ``creat_masks()`` can be analysed with the ``plot_comparison_histogram()`` method which uses ``filter_data()`` internally. This will plot a histogram of the sizes of contiguous areas below a given size limit present in the binary masks before and after applying the retinal vessel annotation skeleton (see the Processing section), and show the mean and 95th percentile in the legend. This serves to show that applying the skeleton indeed reduces the fragmentation of the vessel tree originally caused by thresholding the raw annotation. Isolated areas of annotations are now on average very small and likely caused by involuntary touches of the annotator's pencil, and can therefore be eliminated.


## Splits

For the use in deep learning algorithms, data will usually be split into training and testing subsets. The ``setup_dataset()`` takes care of this, copying images and the required binary ground truth annotation masks into "train" and "test" folders at the desired path. The split can either be specified by passing individual training and testing FOVEA record ids, a fractional data split, or it can left at the default which corresponds to our recommendation. This split was chosen manually to balance retinal conditions, fundus appearance, and image quality between 30 training records and 10 test records.


## Visualisation

To quickly visualise a specific mask, we provide the ``show()`` function which makes use of ``blend_images()`` internally. It employs the OpenCV ``imshow()`` function to display the mask of a given dataset id, annotator number, domain, and annotation type as an overlay over the original image. The ``mask_alpha`` parameter controls mask opacity and is set to 0.2 by default. 0 means only the image is shown, 1 shows only the mask.

As a convenience function, the image is automatically resized to a height of 900 pixels, in order to fit on standard FHD monitors. This can be changed with the optional ``auto_resize`` parameter, or switched off by setting it to 0.


## Reuse

If you use or adapt code from the new files, please cite the FOVEA dataset publication it was written for:

> Ravasio, C and Flores-Sanchez, B and Bloch, E and Bergeles, C and da Cruz, L.
FOVEA: Preoperative and Intraoperative Retinal Fundus Images with Optic Disc and Retinal Vessel Annotations