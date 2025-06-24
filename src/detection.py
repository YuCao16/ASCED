import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    label,
)


def compute_frame_differences(image_sequence):
    differences = []
    for i in range(1, len(image_sequence)):
        diff = np.abs(
            image_sequence[i].astype(np.float32)
            - image_sequence[i - 1].astype(np.float32)
        )
        differences.append(diff)
    return differences


def get_artifact_mask(
    image_sequence,
    mad_scale: float = 3,
    min_area: int = 100,
    max_area: int = 10000,
    min_width: int = 5,
    expand_size: int = 0,
) -> np.ndarray:
    smoothed_diffs = []
    for i in range(1, len(image_sequence)):
        diff = np.abs(
            image_sequence[i].astype(np.float32)
            - image_sequence[i - 1].astype(np.float32)
        )
        smoothed_diffs.append(gaussian_filter(diff, sigma=2))

    artifact_shape = (
        smoothed_diffs[0].shape
        if smoothed_diffs[0].ndim == 2
        else smoothed_diffs[0].shape[:2]
    )
    artifact_mask = np.zeros(artifact_shape, dtype=bool)

    for diff in smoothed_diffs:
        median = np.median(diff)
        mad = np.median(np.abs(diff - median))
        threshold = median + mad_scale * 1.4826 * mad

        diff_max = np.max(diff, axis=2) if diff.ndim == 3 else diff
        artifact_mask |= diff_max > threshold

    labeled_mask, num_features = label(artifact_mask)
    if num_features > 0:
        sizes = np.bincount(labeled_mask.ravel())[1:]
        mask_sizes = np.zeros_like(labeled_mask, dtype=bool)
        for i, size in enumerate(sizes, start=1):
            if min_area < size < max_area:
                mask_sizes[labeled_mask == i] = True
        artifact_mask = mask_sizes

    labeled_mask, num_features = label(artifact_mask)
    filtered_mask = np.zeros_like(artifact_mask, dtype=bool)
    structuring_element = np.ones((min_width, min_width), dtype=bool)

    for region_idx in range(1, num_features + 1):
        region = labeled_mask == region_idx
        eroded_region = binary_erosion(region, structure=structuring_element)
        if eroded_region.sum() > 0:
            filtered_mask |= region

    if expand_size > 0:
        mask_tensor = torch.from_numpy(filtered_mask).bool().float()
        kernel_size = 2 * expand_size + 1
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size), device=mask_tensor.device
        )
        expanded_mask = F.conv2d(
            mask_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=expand_size
        )
        expanded_mask = (expanded_mask > 0).squeeze().numpy()
        return expanded_mask

    return filtered_mask


def get_artifact_mask_args(args, image_sequence):
    return get_artifact_mask(
        image_sequence,
        args.mad_scale,
        args.min_size,
        args.max_size,
        args.narrow_width,
    )
