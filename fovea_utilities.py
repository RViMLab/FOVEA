import shutil
import cv2
import json
import pathlib
import numpy as np
import skimage.morphology
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt


def mask_from_raw(raw_img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Returns a mask containing only the values 0 and 255

    :param raw_img: Input image of shape HW, numpy array of type uint8 with values between 0 and 255
    :param threshold: Threshold fractional value for mask conversion. Between 0 and 1, defaults to 0.5
    :return: Mask as numpy array of shape HW, type uint8, with values either 255 or 0
    """

    mask = (raw_img > threshold * 255).astype('uint8') * 255
    return mask


def skel_from_raw(raw_img: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Returns a skeleton mask containing only the values 0 and 255

    :param raw_img: Input image of shape HW, numpy array of type uint8 with values between 0 and 255
    :param threshold: Threshold fractional value for pixels to be considered. Between 0 and 1, defaults to 0.1
    :return: Skeleton mask as numpy array of shape HW, type uint8, with values either 255 or 0
    """

    mask = (raw_img > threshold * 255).astype('uint8') * 255
    skel = skimage.morphology.skeletonize(mask) * 255
    # skel = cv2.ximgproc.thinning(mask)  # Alternative, needs cv2 contrib installation
    return skel


def find_contiguous_features(mask: np.ndarray) -> (np.ndarray, list):
    """Identifies contiguous features (areas) in a mask and returns their number and size

    :param mask: Input mask as a numpy array of shape HW, with values 0 or 255
    :return: A numpy array of the same shape as the input mask with each contiguous area identified by a unique id
    (0 is background), and a list of the sizes of the contiguous features (areas) as measured in number of pixels
    """

    s = generate_binary_structure(2, 2)
    labels, num_features = label(mask, structure=s)
    feat_sizes = [int(np.sum(labels == (i + 1))) for i in range(num_features)]
    return labels, feat_sizes


def plot_comparison_histogram(path: str, limit: int = 100):
    """Plots a comparative histogram from two lists of feature size lists, includes mean in legend

    :param path: Path to folder containing "feat_sizes.json" created by create_masks() in the "control" folder
        inside of the target directory, if the parameter "control" was set to True.
    :param limit: Upper limit of feature size to consider, defaults to 100
    """

    with open(pathlib.Path(path) / 'feat_sizes.json', 'r') as file:
        stats = json.load(file)
    data1, num1 = filter_data(stats['before'], limit)
    data2, num2 = filter_data(stats['after'], limit)
    bin_width = limit // 10
    bins = range(0, limit + bin_width, bin_width)
    plt.hist(data1, bins=bins, label=f"Before: mean size {int(np.mean(data1))}px, "
                                     f"95th percentile: {int(np.percentile(data1, 95))}px,\n"
                                     f"Mean of {int(np.mean(num1))} features of size < {limit}px per mask")
    plt.hist(data2, bins=bins, label=f"After: mean size {int(np.mean(data2))}px, "
                                     f"95th percentile: {int(np.percentile(data2, 95))}px,\n"
                                     f"Mean of {int(np.mean(num2))} feature of size < {limit}px per mask")
    plt.title(f"Histogram of features of size < {limit}px")
    plt.xlabel("Feature size in pixel count")
    plt.ylabel("Number of features")
    plt.legend()
    plt.show()


def filter_data(feat_sizes: list, limit: int) -> (list, list):
    """Combines list of feature size lists, filtering by an upper limit

    :param feat_sizes: List of feature size lists
    :param limit: Upper limit of feature size to consider
    :return: Flat list of feature size elements below given limit,
    and number of such elements in each feature size list
    """

    data = []
    nums = []
    for sizes in feat_sizes:
        filtered_sizes = list(filter(lambda f: f < limit, sizes))
        data.extend(filtered_sizes)
        nums.append(len(filtered_sizes))
    return data, nums


def prune_features_by_size(
    labels: np.ndarray,
    feat_sizes: list,
    limit: int,
    is_lower_limit: bool = True
) -> np.ndarray:
    """Takes the output of find_contiguous_features() and removes features either below or above a given threshold

    :param labels: Numpy array of shape HW containing contiguous features (areas) labeled by an ID, background being 0
    :param feat_sizes: The sizes of those features, measured in numbers of pixels
    :param limit: Features with a size below this limit will be removed from the mask. If set to -1, the behaviour
    is instead changed to retaining only the largest feature, regardless of size
    :param is_lower_limit: Boolean (defaults to True) determining whether features above (True) or below (False)
    the given limit are preserved. Note: this will not affect the behaviour if limit is set to -1
    :return: Numpy array of the same shape as labels, dtype uint8, 0 on background and 255 on all remaining features
    """

    if limit == -1:
        mask = (labels == np.argmax(feat_sizes) + 1).astype('uint8') * 255
    else:
        mask = np.zeros_like(labels, dtype='uint8')
        for i, size in enumerate(feat_sizes):
            if (is_lower_limit and size >= limit) or (not is_lower_limit and size < limit):
                mask[labels == i + 1] = 255
    return mask


def control_img_from_masks(
    inputs: tuple[np.ndarray],
    cols: tuple = None,
    bg_col: tuple = (255, 255, 255),
) -> np.ndarray:
    """Create a colour image from masks, e.g. to check which mask features are pruned and which are retained

    :param inputs: Tuple of masks as numpy arrays of size HW
    :param cols: Tuple of BGR colour triplets, defaults to ([0, 255, 0], [0, 0, 255], [255, 0, 0])
    :param bg_col: Background colour as BGR colour triplet, defaults to (255, 255, 255) (white)
    :return: Colour image combining the given input masks with given colours, numpy array of shape HW3
    """

    cols = ([0, 255, 0], [0, 0, 255], [255, 0, 0]) if cols is None else cols
    assert len(cols) >= len(inputs), ValueError("Number of colours must match or exceed the number of input masks")
    mask_sum = np.clip(np.add.reduce([i > 0 for i in inputs]), 1, None)
    img = np.zeros(inputs[0].shape + (3,), dtype='f')
    for inp, col in zip(inputs, cols):
        img += (inp[..., None] > 0) * np.array(col)[None, None, :]
        # img[inp > 0] = np.array(col)[None, None, :]
    img /= mask_sum[..., None]
    img[np.sum(img, axis=-1) == 0] = np.array(bg_col)[None, None, :]
    return img.astype('uint8')


def create_masks(
    path: str,
    folder_name: str,
    mask_threshold: float = 0.5,
    skel_threshold: float = 0.1,
    prune_limit: int = 20,
    copy_imgs: bool = True,
    control: bool = True,
):
    """Constructs mask images from the raw annotation data in the given path, and optionally control images

    :param path: Path containing the folder with the raw images
    :param folder_name: Name of folder created inside of given path, will contain mask images. Will overwrite
    :param mask_threshold: Threshold (between 0 and 1) to obtain masks from the raw annotations. Default 0.5, as used
        for the published binary masks
    :param skel_threshold: Threshold (between 0 and 1) for the mask used as a basis for the vessel skeleton.
        Default 0.1, as used for the published binary masks
    :param prune_limit: Size limit below which contiguous features (areas) in the mask are pruned. Default 20,
        as used for the published binary masks
    :param copy_imgs: Boolean, whether images are copied into the new folder containing the new masks. Default True
    :param control: Boolean, default False. If True, control images and data are saved in a folder named "control"
        inside of the given folder_name containing the masks.
    """

    path = pathlib.Path(path)
    ve_feat_sizes = {
        'before': [],
        'after': [],
    }
    (path / folder_name).mkdir(exist_ok=True)
    if control:
        (path / folder_name / 'control').mkdir(exist_ok=True)
    for i in range(1, 41):  # ID
        for domain in ['p', 'i']:  # preop or intraop
            for j in [1, 2]:  # annotator
                # Copy images if required
                if copy_imgs:
                    shutil.copy(path / f'FOVEA{i:03d}_{domain}_img.png', path / folder_name)
                # Load raw annotation
                img = cv2.imread(str(path / f'FOVEA{i:03d}_{domain}_raw_{j}.png'))

                # Vessel masks
                ve_img1 = mask_from_raw(img[..., 2], threshold=mask_threshold)
                skel = skel_from_raw(img[..., 2], threshold=skel_threshold)
                ve_img2 = ve_img1.copy()
                ve_img2[skel > 0] = 255

                labels1, feat_sizes1 = find_contiguous_features(ve_img1)
                labels2, feat_sizes2 = find_contiguous_features(ve_img2)
                ve_feat_sizes['before'].append(feat_sizes1)
                ve_feat_sizes['after'].append(feat_sizes2)
                ve_feat_sizes[f'{i:03d}_{domain}_{j}'] = feat_sizes2

                ve_mask = prune_features_by_size(labels2, feat_sizes2, limit=prune_limit)
                cv2.imwrite(str(path / folder_name / f'FOVEA{i:03d}_{domain}_ve_{j}.png'), ve_mask)

                # Vessel mask control image
                if control:
                    print(f"FOVEA{i:03d}_{domain}_ve_{j}: {feat_sizes2}")
                    pruned_pixels = np.sum(ve_img2 > 0) - np.sum(ve_mask > 0)
                    if pruned_pixels > 0:  # something was pruned
                        ve_mask_pruned = prune_features_by_size(labels2, feat_sizes2,
                                                                limit=prune_limit, is_lower_limit=False)
                        control_img = control_img_from_masks((ve_mask, ve_mask_pruned))
                        cv2.imwrite(str(path / folder_name / 'control' / f'FOVEA{i:03d}_{domain}_ve_{j}'
                                                                         f'__pruned{pruned_pixels}px.png'), control_img)

                # Optical disc masks
                od_img = mask_from_raw(img[..., 1], threshold=mask_threshold)
                labels, feat_sizes = find_contiguous_features(od_img)
                od_mask = prune_features_by_size(labels, feat_sizes, limit=-1)
                cv2.imwrite(str(path / folder_name / f'FOVEA{i:03d}_{domain}_od_{j}.png'), od_mask)

                # Optical disc mask control image
                if control:
                    print(f"FOVEA{i:03d}_{domain}_od_{j}: {feat_sizes}")
                    pruned_pixels = np.sum(od_img > 0) - np.sum(od_mask > 0)
                    if pruned_pixels > 0:  # something was pruned
                        od_mask_pruned = prune_features_by_size(labels, feat_sizes,
                                                                limit=prune_limit, is_lower_limit=False)
                        control_img = control_img_from_masks((od_mask, od_mask_pruned))
                        cv2.imwrite(str(path / folder_name / 'control' / f'FOVEA{i:03d}_{domain}_ve_{j}'
                                                                         f'__pruned{pruned_pixels}px.png'), control_img)

        print(f'Processed FOVEA {i:03d}')

    if control:
        with open(path / folder_name / 'control' / 'feat_sizes.json', 'w', encoding='utf-8') as file:
            json.dump(ve_feat_sizes, file, ensure_ascii=False, indent=4)


def show(
    path: str,
    idx: int,
    annotator: int,
    domain: str,
    mask_type: str,
    mask_alpha: float = 0.2,
    auto_resize: int = 900,
):
    """Helper method that uses cv2.imshow to plot an image/mask overlay

    :param path: Dataset path
    :param idx: ID of the record to be loaded
    :param annotator: Annotator number to be loaded
    :param domain: 'p' for preoperative, 'i' for intraoperative
    :param mask_type: 've' for the vessel mask, 'od' for the optical disc mask
    :param mask_alpha: Blending alpha of the mask. 0 will show only the image, 1 only the mask. Defaults to 0.2
    :param auto_resize: If > 0, the image is resized to height = auto_resize. Defaults to 900 to fit on FHD screens
    """

    path = pathlib.Path(path)
    img = cv2.imread(str(path / f'FOVEA{idx:03d}_{domain}_img.png'))
    mask = cv2.imread(str(path / f'FOVEA{idx:03d}_{domain}_{mask_type}_{annotator}.png'))
    show_img = blend_images(img, mask, alpha=mask_alpha)
    if auto_resize > 0:
        hor_size = int(round(auto_resize * show_img.shape[1] / show_img.shape[0]))
        show_img = cv2.resize(show_img, (hor_size, auto_resize))
    cv2.imshow(f"FOVEA {idx:03d}, {domain}, annotator {annotator}, mask {mask_type}", show_img)
    cv2.waitKey(0)


def blend_images(bot_img: np.ndarray, top_img: np.ndarray, alpha: float) -> np.ndarray:
    """Helper function to blend a top into a bottom image

    :param bot_img: Bottom image as a numpy array of shape HW3
    :param top_img: Top image as a numpy array of shape HW3
    :param alpha: Blending alpha of the top image, 0 will show only the bottom image, 1 only the top image
    :return: Blended image as a numpy array of shape HW3 and type uint8
    """

    assert bot_img.shape == top_img.shape, ValueError("Both images need to have the same height and width")
    img = np.clip(np.round(bot_img * (1 - alpha) + top_img * alpha), 0, 255).astype('uint8')
    return img


def analyse_dataset(path: str):
    """Gathers statistical information on the dataset in the given path and creates some images to evaluate
    annotator agreement:
    - FOVEA<id>_<domain>_od.png: optic disc annotations combined. Green is agreement, blue is annotator 1 only,
        red is annotator 2 only.
    - FOVEA<id>_<domain>_ve.png: vessel annotations combined and overlaid on the original image. Blue is
        annotator 1, green is annotator 2.
    - FOVEA<id>_<domain>_ve.png: vessel annotations shown as skeletons (blue is annotator 1, red is annotator 2)
        and overlaid on the intersection of the full annotations as white. I.e. all white areas are where both
        annotators agree, and the skeleton give a sense of where each annotator interpreted vessel centres to be

    These images and the statistics (as stats.json) are saved in a folder called "stats", inside the existing
    FOVEA dataset folder

    :param path: FOVEA dataset path
    """

    path = pathlib.Path(path)
    (path / 'stats').mkdir()
    ious = {
        'p_ve': [],
        'p_od': [],
        'i_ve': [],
        'i_od': [],
    }
    dices = {
        'p_ve': [],
        'p_od': [],
        'i_ve': [],
        'i_od': [],
    }
    kappas = {
        'p_ve': [],
        'p_od': [],
        'i_ve': [],
        'i_od': [],
    }
    covs = {
        'p_ve_1': [],
        'p_ve_1_skel': [],
        'p_od_1': [],
        'i_ve_1': [],
        'i_ve_1_skel': [],
        'i_od_1': [],
        'p_ve_2': [],
        'p_ve_2_skel': [],
        'p_od_2': [],
        'i_ve_2': [],
        'i_ve_2_skel': [],
        'i_od_2': [],
    }
    for i in range(1, 41):  # ID
        for domain in ['p', 'i']:  # preop or intraop
            #
            mask1 = cv2.imread(str(path / f'FOVEA{i:03d}_{domain}_ve_1.png'), 0)
            mask2 = cv2.imread(str(path / f'FOVEA{i:03d}_{domain}_ve_2.png'), 0)
            img = cv2.imread(str(path / f'FOVEA{i:03d}_{domain}_img.png'))
            control_img = control_img_from_masks((mask1, mask2), cols=((255, 0, 0), (0, 255, 0)), bg_col=(0, 0, 0))
            control_img = blend_images(img, control_img, alpha=0.3)
            cv2.imwrite(str(path / 'stats' / f'FOVEA{i:03d}_{domain}_ve.png'), control_img)
            comb_mask = (mask1 > 0) & (mask2 > 0)
            skel1 = skimage.morphology.skeletonize(mask1)
            skel2 = skimage.morphology.skeletonize(mask2)
            control_img = control_img_from_masks((comb_mask, skel1, skel2),
                                                 ((255, 255, 255), (255, 0, 0), (0, 0, 255)), bg_col=(0, 0, 0))
            cv2.imwrite(str(path / 'stats' / f'FOVEA{i:03d}_{domain}_ve_comb.png'), control_img)
            confusion_matrix = calc_confusion_matrix(mask1, mask2)
            iou, dice, kappa_cohen, coverage1, coverage2 = calc_stats(confusion_matrix)
            skel_cov1 = np.sum(skel1 > 0) / mask1.size
            skel_cov2 = np.sum(skel2 > 0) / mask2.size
            ious[f"{domain}_ve"].append(iou)
            dices[f"{domain}_ve"].append(dice)
            kappas[f"{domain}_ve"].append(kappa_cohen)
            covs[f"{domain}_ve_1"].append(coverage1)
            covs[f"{domain}_ve_2"].append(coverage2)
            covs[f"{domain}_ve_1_skel"].append(skel_cov1)
            covs[f"{domain}_ve_2_skel"].append(skel_cov2)
            # Optical disc
            mask1 = cv2.imread(str(path / f'FOVEA{i:03d}_{domain}_od_1.png'), 0)
            mask2 = cv2.imread(str(path / f'FOVEA{i:03d}_{domain}_od_2.png'), 0)
            img = np.stack(((mask1 == 0) & (mask2 == 0),
                            (mask1 > 0) & (mask2 > 0),
                            (mask1 > 0) & (mask2 == 0),
                            (mask1 == 0) & (mask2 > 0)), axis=0)
            col = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255]])
            img = col[np.argmax(img, axis=0)]
            cv2.imwrite(str(path / 'stats' / f'FOVEA{i:03d}_{domain}_od.png'), img)
            confusion_matrix = calc_confusion_matrix(mask1, mask2)
            iou, dice, kappa_cohen, coverage1, coverage2 = calc_stats(confusion_matrix)
            ious[f"{domain}_od"].append(iou)
            dices[f"{domain}_od"].append(dice)
            kappas[f"{domain}_od"].append(kappa_cohen)
            covs[f"{domain}_od_1"].append(coverage1)
            covs[f"{domain}_od_2"].append(coverage2)
        print(f"Processed FOVEA{i:03d}")
    # Calculate mean values
    means = {}
    for k, vals in ious.items():
        means[f"{k}_mean"] = np.mean(vals)
        print(f"ious {k}: {np.mean(vals):.4f}")
    ious.update(means)
    means = {}
    for k, vals in dices.items():
        means[f"{k}_mean"] = np.mean(vals)
        print(f"dices {k}: {np.mean(vals):.4f}")
    dices.update(means)
    means = {}
    for k, vals in kappas.items():
        means[f"{k}_mean"] = np.mean(vals)
        print(f"kappas {k}: {np.mean(vals):.4f}")
    kappas.update(means)
    means = {}
    for k, vals in covs.items():
        means[f"{k}_mean"] = np.mean(vals)
        print(f"covs {k}: {np.mean(vals):.4f}")
    covs.update(means)
    # Save values
    with open(path / 'stats' / 'stats.json', 'w', encoding='utf-8') as json_file:
        json_dict = {
            'ious': ious,
            'dices': dices,
            'covs': covs,
            'kappas': kappas
        }
        json.dump(json_dict, json_file, ensure_ascii=False, indent=4)


def calc_confusion_matrix(mask1: np.ndarray, mask2: np.ndarray) -> (np.ndarray, float):
    """Calculates the confusion matrix from two masks

    :param mask1: First mask, numpy array of shape HW
    :param mask2: Second mask, numpy array of shape HW
    :return: Confusion matrix as a 2 by 2 numpy array
    """

    #  - confusion matrix between the two:
    #          | cat1 | cat2 |
    #           --------------
    #     cat1 |  A   |  B   |
    #     cat2 |  C   |  D   |
    #
    # A = Annotated by both A1 and A2
    # B = Annotated by A2 but not A1
    # C = Annotated by A1 but not A2
    # D = Annotated by neither A1 nor A2
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    confusion_matrix = np.array(
        [[np.sum(np.logical_and(mask1, mask2)), np.sum(np.logical_and(~mask1, mask2))],
         [np.sum(np.logical_and(mask1, ~mask2)), np.sum(np.logical_and(~mask1, ~mask2))]]
    )
    assert mask1.size == np.sum(confusion_matrix)
    return confusion_matrix


def calc_stats(confusion_matrix: np.ndarray) -> [float, float, float, float]:
    """Calculates a number of stats from a given confusion matrix

    :param confusion_matrix: Confusion matrix for two masks
    :return: IoU, DICE, and the class coverage as a percentage of all pixels for both masks
    """

    #  - confusion matrix between the two:
    #          | cat1 | cat2 |
    #           --------------
    #     cat1 |  A   |  B   |
    #     cat2 |  C   |  D   |
    #
    # A = Annotated by both A1 and A2
    # B = Annotated by A2 but not A1
    # C = Annotated by A1 but not A2
    # D = Annotated by neither A1 nor A2
    c = confusion_matrix / np.sum(confusion_matrix)
    [[a, b],
     [c, d]] = c
    iou = a / (a + c + b)
    dice = 2 * a / (a + c + a + b)
    coverage1 = a + c
    coverage2 = a + b
    p = a + d
    pe = coverage1 * coverage2 + (c + d) * (b + d)
    kappa_cohen = (p - pe) / (1 - pe)
    return iou, dice, kappa_cohen, coverage1, coverage2


def setup_dataset(
    name: str,
    source_folder: str,
    destination_folder: str,
    domain: str,
    annotator: int,
    annotation: tuple[str] = None,
    train_ids: list[int] = None,
    test_ids: list[int] = None,
    split: list[float] = None,
):
    """Automatically copies masks of the given domain, type and annotator into a new folder, split into
    "test" and "train" according to either given id numbers, a given data split, or our suggested 30/10
    split.

    :param name: Name of the dataset split folder. Will contain a "train" and a "test" folder.
    :param source_folder: Path to the FOVEA dataset containing the images and ground truth masks
    :param destination_folder: Path where the split dataset is to be saved at
    :param domain: "i" for intraoperative, "p" for preoperative
    :param annotator: Whose annotator to choose the masks from, 1 or 2
    :param annotation: A list containing the strings "ve" (retinal vessels), "od" (optic disc), or both. Defaults
        to ["ve"], i.e. retinal vessel annotations
    :param train_ids: List of FOVEA ids to use for the training set. Defaults to our suggested 30/10 data split
    :param test_ids: List of FOVEA ids to use for the test set. Must not overlap with the training set. Defaults to
        all ids not contained in train_ids
    :param split: A list containing two floats corresponding to the fractional training / testing data splits.
        Alternative to giving individual train or test ids, will override all previous arguments.
    """

    annotation = ['ve'] if annotation is None else annotation
    # Set up folders
    src_path = pathlib.Path(source_folder)
    path = pathlib.Path(destination_folder) / name
    path.mkdir(exist_ok=False)
    # Set up train / test ids
    if train_ids is None:
        id_lists = {
            'train': [1, 2, 3, 4, 5, 6, 8, 9, 10,
                      11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 29,
                      31, 34, 35, 36],
            'test': [7, 12, 26, 30, 32, 33, 37, 38, 39, 40],
        }
    else:
        id_lists = {'train': train_ids}
        if test_ids is None:
            id_lists['test'] = [i for i in range(1, 41) if i not in train_ids]
        else:
            id_lists['test'] = test_ids
    if split is not None:
        ids = list(range(1, 41))
        train_split = int(40 * split[0])
        test_split = min(int(40 * split[1]), 40 - train_split)
        id_lists = {'train': ids[:train_split], 'test': ids[-test_split:]}
        print(f"Automatic data split used. Actual number of elements: "
              f"train {len(id_lists['train'])}, test {len(id_lists['test'])}")
    if set(id_lists['train']).intersection(set(id_lists['test'])):
        raise ValueError("Some record ids are present in both the training and test set!")
    print(f"Training set: {id_lists['train']}\n    Test set: {id_lists['test']}")
    # Copy files into folders
    for stage, id_list in id_lists.items():  # stage is "train" or "test"
        (path / stage).mkdir()
        for idx in id_list:
            shutil.copy(src_path / f'FOVEA{idx:03d}_{domain}_img.png', path / stage)
            for a in annotation:
                shutil.copy(src_path / f'FOVEA{idx:03d}_{domain}_{a}_{annotator}.png', path / stage)


if __name__ == "__main__":
    path = ""  # Insert path to FOVEA dataset here
    show(path, 1, 1, 'p', 've')
