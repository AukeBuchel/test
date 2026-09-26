from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from .. import measure
from ..filters import rank_order
from .._shared.utils import remove_arg


def _get_high_intensity_peaks(image, mask, num_peaks):
    """
    Return the highest intensity peak coordinates.
    """
    # get coordinates of peaks
    coord = np.nonzero(mask)
    intensities = image[coord]
    # Highest peak first
    idx_maxsort = np.argsort(-intensities)
    coord = np.stack(coord, axis=-1)[idx_maxsort]
    # select num_peaks peaks
    if len(coord) > num_peaks:
        coord = coord[:num_peaks]
    return coord


def _get_peak_mask(image, min_distance, footprint, threshold_abs,
                   threshold_rel):
    """
    Return the mask containing all peak candidates above thresholds.
    """
    if footprint is not None:
        image_max = ndi.maximum_filter(image, footprint=footprint,
                                       mode='constant')
    else:
        size = 2 * min_distance + 1
        image_max = ndi.maximum_filter(image, size=size, mode='constant')
    mask = image == image_max
    if threshold_rel is not None:
        threshold = max(threshold_abs, threshold_rel * image.max())
    else:
        threshold = threshold_abs
    mask &= image > threshold
    return mask


def _ensure_spacing(coord, image, min_distance):
    """
    Enforce min_distance spacing constraint by suppressing peaks that are too close.
    
    For peaks with different intensities, higher intensity peaks suppress lower ones
    within min_distance. For peaks with equal intensities that are within min_distance,
    they are grouped into connected components and one representative is kept per component.
    """
    if len(coord) == 0:
        return coord
    
    if len(coord) == 1:
        return coord
    
    # Get intensities for all coordinates
    intensities = image[tuple(coord.T)]
    
    # Group by intensity level
    unique_intensities = np.unique(intensities)[::-1]  # Highest first
    
    output = []
    suppressed = np.zeros(len(coord), dtype=bool)
    
    for intensity_level in unique_intensities:
        # Get all peaks at this intensity level
        level_mask = intensities == intensity_level
        level_indices = np.where(level_mask)[0]
        level_coords = coord[level_indices]
        
        # Check which peaks at this level are already suppressed by higher-intensity peaks
        for idx in level_indices:
            if suppressed[idx]:
                continue
                
            current_peak = coord[idx]
            
            # Check if suppressed by any already-accepted higher-intensity peak
            is_suppressed = False
            for accepted_peak in output:
                if np.max(np.abs(current_peak - accepted_peak)) < min_distance:
                    is_suppressed = True
                    break
            
            if is_suppressed:
                suppressed[idx] = True
        
        # For remaining peaks at this level, find connected components
        # (peaks within min_distance form components)
        remaining_level_indices = [i for i in level_indices if not suppressed[i]]
        
        if len(remaining_level_indices) == 0:
            continue
        
        remaining_coords = coord[remaining_level_indices]
        
        # Use connected components to group peaks that are close together
        n_remaining = len(remaining_coords)
        adjacency = np.zeros((n_remaining, n_remaining), dtype=bool)
        
        for i in range(n_remaining):
            for j in range(i + 1, n_remaining):
                dist = np.max(np.abs(remaining_coords[i] - remaining_coords[j]))
                if dist < min_distance:
                    adjacency[i, j] = True
                    adjacency[j, i] = True
        
        # Find connected components using simple flood fill
        component_labels = -np.ones(n_remaining, dtype=int)
        current_component = 0
        
        for i in range(n_remaining):
            if component_labels[i] == -1:
                # Start new component
                stack = [i]
                component_labels[i] = current_component
                
                while stack:
                    node = stack.pop()
                    for neighbor in range(n_remaining):
                        if adjacency[node, neighbor] and component_labels[neighbor] == -1:
                            component_labels[neighbor] = current_component
                            stack.append(neighbor)
                
                current_component += 1
        
        # Keep one representative from each component (lexicographically first)
        for component_id in range(current_component):
            component_mask = component_labels == component_id
            component_coords_indices = np.where(component_mask)[0]
            component_coords = remaining_coords[component_coords_indices]
            
            # Pick the lexicographically first coordinate
            lex_order = np.lexsort(component_coords.T[::-1])
            representative_local_idx = lex_order[0]
            representative_idx = remaining_level_indices[component_coords_indices[representative_local_idx]]
            
            output.append(coord[representative_idx])
            
            # Mark all others in this component as suppressed
            for local_idx in component_coords_indices:
                if local_idx != representative_local_idx:
                    global_idx = remaining_level_indices[local_idx]
                    suppressed[global_idx] = True
    
    return np.array(output) if output else np.empty((0, coord.shape[1]), dtype=coord.dtype)


def _exclude_border(mask, exclude_border):
    """
    Remove peaks near the borders
    """
    # zero out the image borders
    for i, excluded in enumerate(exclude_border):
        if excluded == 0:
            continue
        mask[(slice(None),) * i + (slice(None, excluded),)] = False
        mask[(slice(None),) * i + (slice(-excluded, None),)] = False
    return mask


@remove_arg("indices", changed_version="0.20")
def peak_local_max(image, min_distance=1, threshold_abs=None,
                   threshold_rel=None, exclude_border=True, indices=True,
                   num_peaks=np.inf, footprint=None, labels=None,
                   num_peaks_per_label=np.inf):
    """Find peaks in an image as coordinate list or boolean mask.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    If there are multiple local maxima with identical pixel intensities
    inside the region defined with `min_distance`,
    the coordinates of all such pixels are returned.

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int, optional
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.
    threshold_abs : float, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float, optional
        Minimum intensity of peaks, calculated as `max(image) * threshold_rel`.
    exclude_border : int, tuple of ints, or bool, optional
        If positive integer, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If tuple of non-negative ints, the length of the tuple must match the
        input array's dimensionality.  Each element of the tuple will exclude
        peaks from within `exclude_border`-pixels of the border of the image
        along that dimension.
        If True, takes the `min_distance` parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.
    indices : bool, optional
        If True, the output will be an array representing peak
        coordinates. The coordinates are sorted according to peaks
        values (Larger first). If False, the output will be a boolean
        array shaped as `image.shape` with peaks present at True
        elements. ``indices`` is deprecated and will be removed in
        version 0.20. Default behavior will be to always return peak
        coordinates. You can obtain a mask as shown in the example
        below.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`.  Overrides
        `min_distance`.
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.
    num_peaks_per_label : int, optional
        Maximum number of peaks for each label.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. A maximum filter is used for finding local maxima.
    This operation dilates the original image. After comparison of the dilated
    and original image, this function returns the coordinates or a mask of the
    peaks where the dilated image equals the original image.

    See also
    --------
    skimage.feature.corner_peaks

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 1.5, 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> peak_idx = peak_local_max(img2, exclude_border=0)
    >>> peak_idx
    array([[10, 10, 10]])

    >>> peak_mask = np.zeros_like(img2, dtype=bool)
    >>> peak_mask[peak_idx] = True

    """
    out = np.zeros_like(image, dtype=np.bool)

    threshold_abs = threshold_abs if threshold_abs is not None else image.min()

    if min_distance < 1:
        warn("When min_distance < 1, peak_local_max acts as finding "
             "image > max(threshold_abs, threshold_rel).",
             RuntimeWarning, stacklevel=2)

    if footprint is not None:
        footprint = np.asarray(footprint)
        if footprint.size == 1:
            warn("When footprint.size < 2, peak_local_max acts as finding "
                 "image > max(threshold_abs, threshold_rel).",
                 RuntimeWarning, stacklevel=2)

    if isinstance(exclude_border, bool):
        exclude_border = (min_distance if exclude_border else 0,) * image.ndim
    elif isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError("`exclude_border` cannot be a negative value")
        exclude_border = (exclude_border,) * image.ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != image.ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image.")
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "`exclude_border`, when expressed as a tuple, must only "
                    "contain ints."
                )
            if exclude < 0:
                raise ValueError(
                    "`exclude_border` cannot contain a negative value")
    else:
        raise TypeError(
            "`exclude_border` must be bool, int, or tuple with the same "
            "length as the dimensionality of the image.")

    # no peak for a trivial image
    if np.all(image == image.flat[0]):
        if indices is True:
            return np.empty((0, image.ndim), np.int)
        else:
            return out

    # In the case of labels, call ndi on each label
    if labels is not None:
        # Create a copy to avoid modifying the caller's input
        labels = labels.copy()
        label_values = np.unique(labels)
        # Reorder label values to have consecutive integers (no gaps)
        if np.any(np.diff(label_values) != 1):
            mask = labels >= 1
            labels[mask] = 1 + rank_order(labels[mask])[0].astype(labels.dtype)
        labels = labels.astype(np.int32)

        # create a mask for the non-exclude region
        inner_mask = _exclude_border(np.ones_like(labels, dtype=bool),
                                     exclude_border)

        # For each label, extract a smaller image enclosing the object of
        # interest, identify num_peaks_per_label peaks and mark them in
        # variable out.
        for label_idx, obj in enumerate(ndi.find_objects(labels)):
            img_object = image[obj] * (labels[obj] == label_idx + 1)
            mask = _get_peak_mask(img_object, min_distance, footprint,
                                  threshold_abs, threshold_rel)
            if exclude_border:
                # remove peaks fall in the exclude region
                mask &= inner_mask[obj]
            coordinates = _get_high_intensity_peaks(img_object, mask,
                                                    num_peaks_per_label)
            # Enforce min_distance spacing within this label
            if min_distance > 0 and len(coordinates) > 0:
                coordinates = _ensure_spacing(coordinates, img_object, min_distance)
            nd_indices = tuple(coordinates.T)
            mask.fill(False)
            mask[nd_indices] = True
            out[obj] += mask

        if not indices and np.isinf(num_peaks):
            return out

        coordinates = _get_high_intensity_peaks(image, out, num_peaks)
        if indices:
            return coordinates
        else:
            out.fill(False)
            nd_indices = tuple(coordinates.T)
            out[nd_indices] = True
            return out

    # Non maximum filter
    mask = _get_peak_mask(image, min_distance, footprint, threshold_abs,
                          threshold_rel)

    mask = _exclude_border(mask, exclude_border)

    # Select highest intensities (num_peaks)
    coordinates = _get_high_intensity_peaks(image, mask, num_peaks)
    
    # Enforce min_distance spacing
    if min_distance > 0 and len(coordinates) > 0:
        coordinates = _ensure_spacing(coordinates, image, min_distance)

    if indices is True:
        return coordinates
    else:
        nd_indices = tuple(coordinates.T)
        out[nd_indices] = True
        return out


def _prominent_peaks(image, min_xdistance=1, min_ydistance=1,
                     threshold=None, num_peaks=np.inf):
    """Return peaks with non-maximum suppression.

    Identifies most prominent features separated by certain distances.
    Non-maximum suppression with different sizes is applied separately
    in the first and second dimension of the image to identify peaks.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    min_xdistance : int
        Minimum distance separating features in the x dimension.
    min_ydistance : int
        Minimum distance separating features in the y dimension.
    threshold : float
        Minimum intensity of peaks. Default is `0.5 * max(image)`.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    intensity, xcoords, ycoords : tuple of array
        Peak intensity values, x and y indices.
    """

    img = image.copy()
    rows, cols = img.shape

    if threshold is None:
        threshold = 0.5 * np.max(img)

    ycoords_size = 2 * min_ydistance + 1
    xcoords_size = 2 * min_xdistance + 1
    img_max = ndi.maximum_filter1d(img, size=ycoords_size, axis=0,
                                   mode='constant', cval=0)
    img_max = ndi.maximum_filter1d(img_max, size=xcoords_size, axis=1,
                                   mode='constant', cval=0)
    mask = (img == img_max)
    img *= mask
    img_t = img > threshold

    label_img = measure.label(img_t)
    props = measure.regionprops(label_img, img_max)

    # Sort the list of peaks by intensity, not left-right, so larger peaks
    # in Hough space cannot be arbitrarily suppressed by smaller neighbors
    props = sorted(props, key=lambda x: x.max_intensity)[::-1]
    coords = np.array([np.round(p.centroid) for p in props], dtype=int)

    img_peaks = []
    ycoords_peaks = []
    xcoords_peaks = []

    # relative coordinate grid for local neighbourhood suppression
    ycoords_ext, xcoords_ext = np.mgrid[-min_ydistance:min_ydistance + 1,
                                        -min_xdistance:min_xdistance + 1]

    for ycoords_idx, xcoords_idx in coords:
        accum = img_max[ycoords_idx, xcoords_idx]
        if accum > threshold:
            # absolute coordinate grid for local neighbourhood suppression
            ycoords_nh = ycoords_idx + ycoords_ext
            xcoords_nh = xcoords_idx + xcoords_ext

            # no reflection for distance neighbourhood
            ycoords_in = np.logical_and(ycoords_nh > 0, ycoords_nh < rows)
            ycoords_nh = ycoords_nh[ycoords_in]
            xcoords_nh = xcoords_nh[ycoords_in]

            # reflect xcoords and assume xcoords are continuous,
            # e.g. for angles:
            # (..., 88, 89, -90, -89, ..., 89, -90, -89, ...)
            xcoords_low = xcoords_nh < 0
            ycoords_nh[xcoords_low] = rows - ycoords_nh[xcoords_low]
            xcoords_nh[xcoords_low] += cols
            xcoords_high = xcoords_nh >= cols
            ycoords_nh[xcoords_high] = rows - ycoords_nh[xcoords_high]
            xcoords_nh[xcoords_high] -= cols

            # suppress neighbourhood
            img_max[ycoords_nh, xcoords_nh] = 0

            # add current feature to peaks
            img_peaks.append(accum)
            ycoords_peaks.append(ycoords_idx)
            xcoords_peaks.append(xcoords_idx)

    img_peaks = np.array(img_peaks)
    ycoords_peaks = np.array(ycoords_peaks)
    xcoords_peaks = np.array(xcoords_peaks)

    if num_peaks < len(img_peaks):
        idx_maxsort = np.argsort(img_peaks)[::-1][:num_peaks]
        img_peaks = img_peaks[idx_maxsort]
        ycoords_peaks = ycoords_peaks[idx_maxsort]
        xcoords_peaks = xcoords_peaks[idx_maxsort]

    return img_peaks, xcoords_peaks, ycoords_peaks
