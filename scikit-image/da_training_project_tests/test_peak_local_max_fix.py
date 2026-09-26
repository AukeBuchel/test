"""
Test suite for the peak_local_max bug fixes described in problem.txt.

Tests cover:
- min_distance enforcement (including equal-valued plateaus)
- labels array non-mutation
- labels with non-consecutive integer values
- Preserved existing behavior (num_peaks, threshold_abs, threshold_rel,
  exclude_border, suppress lower-intensity peaks, N-dimensional support)
- Public function signature unchanged
"""
import inspect
import numpy as np
import pytest
from skimage.feature import peak_local_max


def _chebyshev(a, b):
    """Chebyshev (L-infinity) distance between two coordinate arrays."""
    return np.max(np.abs(np.asarray(a) - np.asarray(b)))


def _assert_min_distance(coords, min_distance):
    """Assert every pair of returned coordinates respects min_distance."""
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = _chebyshev(coords[i], coords[j])
            assert dist >= min_distance, (
                f"Peaks {coords[i]} and {coords[j]} are only {dist} apart, "
                f"but min_distance={min_distance}"
            )


# -----------------------------------------------------------------------
# 1. min_distance enforcement on equal-valued plateau (the primary bug)
# -----------------------------------------------------------------------

class TestMinDistancePlateau:
    """Verify min_distance is enforced when the image contains a flat plateau
    of equal-valued pixels that all qualify as local maxima."""

    def test_horizontal_plateau_min_distance_10(self):
        """Reproduction case from the problem statement."""
        image = np.zeros((30, 60))
        image[15, 20:41] = 5.0
        coords = peak_local_max(image, min_distance=10)
        assert len(coords) >= 1, "Plateau region must contribute at least one peak"
        _assert_min_distance(coords, 10)

    def test_horizontal_plateau_min_distance_5(self):
        """Shorter min_distance should allow more peaks from the same plateau."""
        image = np.zeros((30, 60))
        image[15, 20:41] = 5.0
        coords = peak_local_max(image, min_distance=5)
        assert len(coords) >= 1
        _assert_min_distance(coords, 5)

    def test_2d_block_plateau(self):
        """A 2D block of equal-valued pixels."""
        image = np.zeros((30, 30))
        image[5:16, 5:16] = 3.0  # 11x11 block
        coords = peak_local_max(image, min_distance=5)
        assert len(coords) >= 1, "Block must produce at least one peak"
        _assert_min_distance(coords, 5)

    def test_small_plateau_min_distance_1(self):
        """A 2x2 block with min_distance=1; all four pixels are exactly
        min_distance apart in Chebyshev metric, so all four should be kept."""
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1:3, 1:3] = 10
        coords = peak_local_max(image, min_distance=1)
        assert len(coords) == 4

    def test_large_plateau_returns_at_least_one(self):
        """Even a very large plateau must yield at least one coordinate."""
        image = np.ones((50, 50)) * 10
        image[0, :] = 0  # ensure it's not a constant image
        coords = peak_local_max(image, min_distance=20)
        assert len(coords) >= 1

    def test_plateau_with_higher_neighbor(self):
        """A plateau adjacent to a strictly higher peak: the plateau should be
        suppressed by the higher peak if within min_distance."""
        image = np.zeros((40, 40))
        image[20, 10:17] = 5.0  # plateau of 7 pixels
        image[20, 25] = 10.0    # higher isolated peak
        coords = peak_local_max(image, min_distance=8)
        _assert_min_distance(coords, 8)
        intensities = image[tuple(coords.T)]
        assert 10.0 in intensities, "Higher peak must always be present"


# -----------------------------------------------------------------------
# 2. Existing correct behavior: lower intensity suppressed by higher
# -----------------------------------------------------------------------

class TestLowerIntensitySuppression:
    """Strictly lower-intensity candidates within min_distance of a strictly
    higher one must remain suppressed."""

    def test_problem_statement_contrast_case(self):
        """Second reproduction script from the problem statement."""
        image = np.zeros((30, 60))
        image[15, 30] = 5.0
        image[15, 32] = 3.0
        coords = peak_local_max(image, min_distance=10)
        assert len(coords) == 1
        assert list(coords[0]) == [15, 30]

    def test_two_peaks_different_intensity(self):
        """Higher peak survives, lower within min_distance is suppressed."""
        image = np.zeros((20, 20))
        image[10, 10] = 10.0
        image[10, 12] = 5.0
        coords = peak_local_max(image, min_distance=5)
        assert len(coords) == 1
        assert list(coords[0]) == [10, 10]

    def test_two_peaks_far_apart_both_kept(self):
        """When peaks are far enough apart, both should be kept regardless of
        intensity difference."""
        image = np.zeros((20, 40))
        image[10, 10] = 10.0
        image[10, 30] = 3.0
        coords = peak_local_max(image, min_distance=5)
        assert len(coords) == 2
        _assert_min_distance(coords, 5)


# -----------------------------------------------------------------------
# 3. Labels: non-mutation of caller's array
# -----------------------------------------------------------------------

class TestLabelsNonMutation:
    """The caller's labels array must never be modified by peak_local_max."""

    def test_problem_statement_labels_mutation(self):
        """Reproduction case from the problem statement."""
        img = np.zeros((10, 20))
        img[3, 3] = 4.0
        img[5, 15] = 4.0
        labels = np.zeros((10, 20), int)
        labels[2:5, 2:5] = 2
        labels[4:7, 13:17] = 5
        before = labels.copy()
        peak_local_max(img, labels=labels, min_distance=1)
        np.testing.assert_array_equal(labels, before)

    def test_labels_with_large_gap(self):
        """Non-consecutive label values with a large gap (triggers rank_order
        renumbering internally)."""
        img = np.zeros((10, 20))
        img[3, 3] = 4.0
        img[8, 15] = 4.0
        labels = np.zeros((10, 20), int)
        labels[2:5, 2:5] = 7
        labels[7:10, 14:18] = 1000
        before = labels.copy()
        peak_local_max(img, labels=labels, min_distance=1)
        np.testing.assert_array_equal(labels, before)

    def test_labels_consecutive_no_mutation(self):
        """Even with already-consecutive labels, the array must not be mutated."""
        img = np.zeros((10, 20))
        img[3, 3] = 4.0
        img[8, 15] = 4.0
        labels = np.zeros((10, 20), int)
        labels[2:5, 2:5] = 1
        labels[7:10, 14:18] = 2
        before = labels.copy()
        peak_local_max(img, labels=labels, min_distance=1)
        np.testing.assert_array_equal(labels, before)

    def test_image_array_not_mutated(self):
        """The image array passed by the caller must also be left unchanged."""
        img = np.zeros((10, 20))
        img[3, 3] = 4.0
        img[5, 15] = 4.0
        before = img.copy()
        labels = np.zeros((10, 20), int)
        labels[2:5, 2:5] = 2
        labels[4:7, 13:17] = 5
        peak_local_max(img, labels=labels, min_distance=1)
        np.testing.assert_array_equal(img, before)


# -----------------------------------------------------------------------
# 4. Labels: arbitrary positive integer values
# -----------------------------------------------------------------------

class TestLabelsArbitraryValues:
    """Label values are arbitrary positive integers; zero is background."""

    def test_non_consecutive_labels_find_all_peaks(self):
        """Each labeled region should contribute peaks regardless of the
        specific integer label values used."""
        img = np.zeros((20, 40))
        img[5, 5] = 10.0
        img[5, 35] = 10.0
        img[15, 5] = 10.0
        labels = np.zeros((20, 40), int)
        labels[3:8, 3:8] = 3
        labels[3:8, 33:38] = 99
        labels[13:18, 3:8] = 500
        coords = peak_local_max(img, labels=labels, min_distance=1)
        assert len(coords) == 3

    def test_label_value_one_works(self):
        """A single region with label value 1."""
        img = np.zeros((10, 10))
        img[5, 5] = 10.0
        labels = np.zeros((10, 10), int)
        labels[3:8, 3:8] = 1
        coords = peak_local_max(img, labels=labels, min_distance=1)
        assert len(coords) == 1
        assert list(coords[0]) == [5, 5]

    def test_zero_label_is_background(self):
        """Pixels with label 0 are background and their peaks should not
        appear in the output."""
        img = np.zeros((10, 10))
        img[5, 5] = 10.0  # in background (label 0)
        img[5, 8] = 10.0  # in region label 1
        labels = np.zeros((10, 10), int)
        labels[4:7, 7:10] = 1
        coords = peak_local_max(img, labels=labels, min_distance=1,
                                exclude_border=False)
        assert len(coords) == 1
        assert list(coords[0]) == [5, 8]


# -----------------------------------------------------------------------
# 5. Labels: min_distance enforced within each region
# -----------------------------------------------------------------------

class TestLabelsMinDistancePerRegion:
    """Within each labeled region, min_distance must be enforced. Peaks from
    different regions are independent."""

    def test_plateau_within_label(self):
        """A plateau inside a labeled region should be thinned by min_distance."""
        img = np.zeros((20, 40))
        img[10, 10:25] = 5.0  # plateau of 15 pixels inside label 1
        labels = np.zeros((20, 40), int)
        labels[8:13, 8:27] = 1
        coords = peak_local_max(img, labels=labels, min_distance=5,
                                exclude_border=False)
        assert len(coords) >= 1
        _assert_min_distance(coords, 5)

    def test_cross_region_peaks_within_min_distance(self):
        """Peaks in different label regions that are closer than min_distance
        should both be kept."""
        img = np.zeros((10, 20))
        img[5, 5] = 10.0
        img[5, 7] = 10.0
        labels = np.zeros((10, 20), int)
        labels[3:8, 3:7] = 1
        labels[3:8, 7:10] = 2
        coords = peak_local_max(img, labels=labels, min_distance=5,
                                exclude_border=False)
        assert len(coords) == 2, (
            "Peaks from different regions need not respect min_distance"
        )


# -----------------------------------------------------------------------
# 6. Preserved behavior: num_peaks
# -----------------------------------------------------------------------

class TestNumPeaks:
    def test_num_peaks_limits_output(self):
        image = np.zeros((20, 20))
        image[5, 5] = 10
        image[5, 15] = 9
        image[15, 5] = 8
        image[15, 15] = 7
        coords = peak_local_max(image, min_distance=1, num_peaks=2)
        assert len(coords) == 2
        intensities = sorted(image[tuple(coords.T)], reverse=True)
        assert intensities == [10, 9], (
            "num_peaks should keep the highest-intensity peaks"
        )

    def test_num_peaks_inf_returns_all(self):
        image = np.zeros((20, 20))
        image[5, 5] = 10
        image[5, 15] = 9
        image[15, 5] = 8
        image[15, 15] = 7
        coords = peak_local_max(image, min_distance=1, num_peaks=np.inf)
        assert len(coords) == 4


# -----------------------------------------------------------------------
# 7. Preserved behavior: threshold_abs
# -----------------------------------------------------------------------

class TestThresholdAbs:
    def test_threshold_abs_filters(self):
        image = np.zeros((20, 20))
        image[5, 5] = 10
        image[10, 10] = 5
        image[15, 15] = 2
        coords = peak_local_max(image, min_distance=1, threshold_abs=4)
        intensities = image[tuple(coords.T)]
        assert all(i > 4 for i in intensities)
        assert len(coords) == 2

    def test_threshold_abs_none_uses_image_min(self):
        """When threshold_abs is None the effective threshold is image.min()."""
        image = np.zeros((10, 10))
        image[5, 5] = 1
        coords = peak_local_max(image, min_distance=1, threshold_abs=None)
        assert len(coords) == 1


# -----------------------------------------------------------------------
# 8. Preserved behavior: threshold_rel
# -----------------------------------------------------------------------

class TestThresholdRel:
    def test_threshold_rel_filters(self):
        image = np.zeros((20, 20))
        image[5, 5] = 10
        image[10, 10] = 3
        coords = peak_local_max(image, min_distance=1, threshold_rel=0.5)
        assert len(coords) == 1
        assert list(coords[0]) == [5, 5]

    def test_threshold_rel_and_abs_combined(self):
        """max(threshold_abs, threshold_rel * image.max()) is used."""
        image = np.zeros((20, 20))
        image[5, 5] = 10
        image[10, 10] = 6
        image[15, 15] = 3
        coords = peak_local_max(image, min_distance=1,
                                threshold_abs=5, threshold_rel=0.2)
        # threshold_rel = 0.2 * 10 = 2, threshold_abs = 5, max = 5
        intensities = image[tuple(coords.T)]
        assert all(i > 5 for i in intensities)


# -----------------------------------------------------------------------
# 9. Preserved behavior: exclude_border
# -----------------------------------------------------------------------

class TestExcludeBorder:
    def test_exclude_border_true_uses_min_distance(self):
        image = np.zeros((20, 20))
        image[1, 1] = 10
        image[10, 10] = 10
        coords_true = peak_local_max(image, min_distance=3,
                                     exclude_border=True)
        coords_int = peak_local_max(image, min_distance=3,
                                    exclude_border=3)
        np.testing.assert_array_equal(
            sorted(coords_true.tolist()), sorted(coords_int.tolist())
        )

    def test_exclude_border_false_keeps_edge_peaks(self):
        image = np.zeros((10, 10))
        image[0, 0] = 10
        image[5, 5] = 10
        coords = peak_local_max(image, min_distance=1, exclude_border=False)
        assert len(coords) == 2

    def test_exclude_border_zero_keeps_edge_peaks(self):
        image = np.zeros((10, 10))
        image[0, 0] = 10
        image[5, 5] = 10
        coords = peak_local_max(image, min_distance=1, exclude_border=0)
        assert len(coords) == 2


# -----------------------------------------------------------------------
# 10. N-dimensional support
# -----------------------------------------------------------------------

class TestNDimensional:
    def test_3d_peaks(self):
        image = np.zeros((30, 30, 30))
        image[15, 15, 15] = 1
        image[5, 5, 5] = 1
        coords = peak_local_max(image, min_distance=10, threshold_rel=0,
                                exclude_border=False)
        assert len(coords) == 2
        assert coords.shape[1] == 3
        _assert_min_distance(coords, 10)

    def test_4d_peaks(self):
        image = np.zeros((15, 15, 15, 15))
        image[5, 5, 5, 5] = 1
        image[10, 10, 10, 10] = 1
        coords = peak_local_max(image, min_distance=3, threshold_rel=0,
                                exclude_border=False)
        assert len(coords) == 2
        assert coords.shape[1] == 4
        _assert_min_distance(coords, 3)

    def test_3d_plateau(self):
        """min_distance must also be enforced in 3D plateaus."""
        image = np.zeros((20, 20, 20))
        image[10, 10, 5:16] = 5.0
        coords = peak_local_max(image, min_distance=5, exclude_border=False)
        assert len(coords) >= 1
        _assert_min_distance(coords, 5)


# -----------------------------------------------------------------------
# 11. Public function signature unchanged
# -----------------------------------------------------------------------

class TestFunctionSignature:
    def test_parameter_names_and_order(self):
        sig = inspect.signature(peak_local_max)
        params = list(sig.parameters.keys())
        expected = ['image', 'min_distance', 'threshold_abs', 'threshold_rel',
                    'exclude_border', 'indices', 'num_peaks', 'footprint',
                    'labels', 'num_peaks_per_label']
        assert params == expected

    def test_default_values(self):
        sig = inspect.signature(peak_local_max)
        p = sig.parameters
        assert p['min_distance'].default == 1
        assert p['threshold_abs'].default is None
        assert p['threshold_rel'].default is None
        assert p['exclude_border'].default is True
        assert p['indices'].default is True
        assert p['footprint'].default is None
        assert p['labels'].default is None


# -----------------------------------------------------------------------
# 12. Region coverage: plateau must not be entirely dropped
# -----------------------------------------------------------------------

class TestRegionCoverage:
    def test_single_plateau_not_dropped(self):
        """A single qualifying plateau must contribute at least one peak."""
        image = np.zeros((30, 30))
        image[10:20, 10:20] = 7.0
        coords = peak_local_max(image, min_distance=3)
        assert len(coords) >= 1

    def test_multiple_separated_plateaus(self):
        """Two separated plateaus should each contribute at least one peak."""
        image = np.zeros((30, 70))
        image[15, 15:25] = 5.0
        image[15, 45:55] = 5.0
        coords = peak_local_max(image, min_distance=5)
        assert len(coords) >= 2, "Each plateau must contribute at least one peak"
        _assert_min_distance(coords, 5)

    def test_plateau_larger_than_min_distance_multiple_peaks(self):
        """A region larger than min_distance may legitimately contribute more
        than one coordinate, as long as spacing is respected."""
        image = np.zeros((30, 60))
        image[15, 10:51] = 5.0  # 41 pixels wide
        coords = peak_local_max(image, min_distance=10)
        assert len(coords) >= 1
        _assert_min_distance(coords, 10)

    def test_distinct_peaks_same_intensity_both_kept(self):
        """Two isolated peaks with the same intensity that are far enough
        apart must both be returned."""
        image = np.zeros((20, 40))
        image[10, 10] = 5.0
        image[10, 30] = 5.0
        coords = peak_local_max(image, min_distance=5)
        assert len(coords) == 2
        _assert_min_distance(coords, 5)


# -----------------------------------------------------------------------
# 13. Edge / corner cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_image_returns_empty(self):
        image = np.full((20, 20), 5.0)
        coords = peak_local_max(image, min_distance=1)
        assert len(coords) == 0

    def test_single_pixel_peak(self):
        image = np.zeros((10, 10))
        image[5, 5] = 1
        coords = peak_local_max(image, min_distance=1)
        assert len(coords) == 1
        assert list(coords[0]) == [5, 5]

    def test_all_zeros_returns_empty(self):
        image = np.zeros((10, 10))
        coords = peak_local_max(image, min_distance=1)
        assert len(coords) == 0

    def test_min_distance_equals_1_adjacent_equal_peaks(self):
        """With min_distance=1, peaks exactly 1 apart in Chebyshev distance
        are at exactly min_distance and should be retained."""
        image = np.zeros((10, 10))
        image[5, 5] = 10
        image[5, 6] = 10
        coords = peak_local_max(image, min_distance=1)
        assert len(coords) == 2

    def test_output_is_ndarray(self):
        image = np.zeros((10, 10))
        image[5, 5] = 1
        coords = peak_local_max(image, min_distance=1)
        assert isinstance(coords, np.ndarray)

    def test_output_shape_columns_match_ndim(self):
        for ndim in (2, 3):
            shape = (10,) * ndim
            image = np.zeros(shape)
            center = (5,) * ndim
            image[center] = 1
            coords = peak_local_max(image, min_distance=1)
            assert coords.shape[1] == ndim

# -----------------------------------------------------------------------
# 14. Missing coverage
# -----------------------------------------------------------------------

class TestChebyshevMetric:
 
    def test_diagonal_pair_inside_min_distance_collapses(self):
        """Chebyshev distance 3 < min_distance 4 -> must collapse to one peak.
        A Euclidean implementation sees 4.24 >= 4 and wrongly keeps both."""
        image = np.zeros((30, 30))
        image[10, 10] = 5.0
        image[13, 13] = 5.0
        coords = peak_local_max(image, min_distance=4, exclude_border=False)
        assert len(coords) == 1
        _assert_min_distance(coords, 4)
 
    def test_diagonal_pair_at_exactly_min_distance_both_kept(self):
        """Chebyshev distance 4 == min_distance 4 -> 'at least min_distance
        apart' is satisfied, so both must be kept. Guards against a metric
        that is too aggressive."""
        image = np.zeros((30, 30))
        image[10, 10] = 5.0
        image[14, 14] = 5.0
        coords = peak_local_max(image, min_distance=4, exclude_border=False)
        assert len(coords) == 2
        _assert_min_distance(coords, 4)
 
    def test_diagonal_block_plateau_chebyshev_spacing(self):
        """A 2-D tied block: every returned pair must satisfy Chebyshev
        spacing, including diagonal pairs."""
        image = np.zeros((40, 40))
        image[10:25, 10:25] = 5.0
        coords = peak_local_max(image, min_distance=6, exclude_border=False)
        assert len(coords) >= 1
        _assert_min_distance(coords, 6)
 
    def test_3d_diagonal_pair_collapses(self):
        """Same metric requirement in 3-D: Chebyshev 3 < 5 -> collapse."""
        image = np.zeros((20, 20, 20))
        image[5, 5, 5] = 5.0
        image[8, 8, 8] = 5.0
        coords = peak_local_max(image, min_distance=5, exclude_border=False)
        assert len(coords) == 1
        
class TestNumPeaksWithTies:
 
    def _five_tied_plateaus(self):
        image = np.zeros((30, 200))
        for s in (5, 40, 75, 110, 145):
            image[15, s:s + 8] = 5.0
        return image
 
    def test_all_tied_plateaus_resolve_to_one_each(self):
        image = self._five_tied_plateaus()
        coords = peak_local_max(image, min_distance=10)
        assert len(coords) == 5
        _assert_min_distance(coords, 10)
 
    def test_num_peaks_caps_resolved_peaks_not_raw_candidates(self):
        """Five resolved peaks exist and num_peaks=3, so exactly 3 must be
        returned. An implementation that truncates to num_peaks BEFORE
        de-duplicating returns fewer than 3."""
        image = self._five_tied_plateaus()
        coords = peak_local_max(image, min_distance=10, num_peaks=3)
        assert len(coords) == 3
        _assert_min_distance(coords, 10)
 
    def test_num_peaks_one_with_ties(self):
        image = self._five_tied_plateaus()
        coords = peak_local_max(image, min_distance=10, num_peaks=1)
        assert len(coords) == 1
 
    def test_num_peaks_larger_than_available_returns_all(self):
        image = self._five_tied_plateaus()
        coords = peak_local_max(image, min_distance=10, num_peaks=50)
        assert len(coords) == 5


class TestNumPeaksPerLabelWithTies:
 
    def _two_regions_with_wide_plateaus(self):
        image = np.zeros((30, 200))
        labels = np.zeros((30, 200), dtype=int)
        labels[:, 0:100] = 1
        labels[:, 100:200] = 2
        image[15, 10:51] = 5.0     # wide tied plateau in region 1
        image[15, 110:151] = 5.0   # wide tied plateau in region 2
        return image, labels
 
    def test_num_peaks_per_label_caps_resolved_peaks(self):
        """Each region's wide plateau resolves to several spaced peaks;
        num_peaks_per_label=2 must yield exactly 2 per region (4 total)."""
        image, labels = self._two_regions_with_wide_plateaus()
        coords = peak_local_max(image, labels=labels, min_distance=10,
                                num_peaks_per_label=2, exclude_border=False)
        assert len(coords) == 4
        per_region = [labels[tuple(c)] for c in coords]
        assert per_region.count(1) == 2
        assert per_region.count(2) == 2
 
    def test_num_peaks_per_label_one(self):
        image, labels = self._two_regions_with_wide_plateaus()
        coords = peak_local_max(image, labels=labels, min_distance=10,
                                num_peaks_per_label=1, exclude_border=False)
        assert len(coords) == 2
        assert {int(labels[tuple(c)]) for c in coords} == {1, 2}
        
        
class TestExcludeBorderTuple:
    def test_tuple_excludes_first_axis_only(self):
        image = np.zeros((10, 10))
        image[0, 5] = 10.0   # on the first-axis border
        image[5, 0] = 10.0   # on the second-axis border
        coords = peak_local_max(image, min_distance=1, exclude_border=(1, 0))
        found = {tuple(c) for c in coords}
        assert (0, 5) not in found, "first-axis border peak must be excluded"
        assert (5, 0) in found, "second-axis border peak must be kept"
 
    def test_tuple_excludes_second_axis_only(self):
        image = np.zeros((10, 10))
        image[0, 5] = 10.0
        image[5, 0] = 10.0
        coords = peak_local_max(image, min_distance=1, exclude_border=(0, 1))
        found = {tuple(c) for c in coords}
        assert (0, 5) in found, "first-axis border peak must be kept"
        assert (5, 0) not in found, "second-axis border peak must be excluded"
 
    def test_tuple_wrong_cardinality_raises(self):
        image = np.zeros((5, 5))
        with pytest.raises(ValueError):
            peak_local_max(image, exclude_border=(1,))
 
    def test_tuple_non_integer_raises(self):
        image = np.zeros((5, 5))
        with pytest.raises(ValueError):
            peak_local_max(image, exclude_border=(1, 'a'))
 
    def test_tuple_negative_raises(self):
        image = np.zeros((5, 5))
        with pytest.raises(ValueError):
            peak_local_max(image, exclude_border=(1, -1))
 
    def test_negative_int_raises(self):
        image = np.zeros((5, 5))
        with pytest.raises(ValueError):
            peak_local_max(image, exclude_border=-1)
 
    def test_float_raises_type_error(self):
        image = np.zeros((5, 5))
        with pytest.raises(TypeError):
            peak_local_max(image, exclude_border=1.0)

 
class TestIndicesFalseMaskPath:
 
    def test_mask_matches_coordinates_on_plateau(self):
        image = np.zeros((30, 60))
        image[15, 20:41] = 5.0
        coords = peak_local_max(image, min_distance=10)
        mask = peak_local_max(image, min_distance=10, indices=False)
        assert mask.shape == image.shape
        assert mask.dtype == bool
        assert mask.sum() == len(coords)
        assert {tuple(c) for c in coords} == {tuple(p) for p in np.argwhere(mask)}
 
    def test_mask_respects_min_distance(self):
        image = np.zeros((30, 60))
        image[15, 20:41] = 5.0
        mask = peak_local_max(image, min_distance=10, indices=False)
        _assert_min_distance(np.argwhere(mask), 10)
 
    def test_mask_with_labels(self):
        image = np.zeros((20, 40))
        image[10, 10:25] = 5.0
        labels = np.zeros((20, 40), dtype=int)
        labels[8:13, 8:27] = 1
        coords = peak_local_max(image, labels=labels, min_distance=5,
                                exclude_border=False)
        mask = peak_local_max(image, labels=labels, min_distance=5,
                              exclude_border=False, indices=False)
        assert mask.sum() == len(coords)
        _assert_min_distance(np.argwhere(mask), 5)
        
class TestFootprintPreserved:
    def test_footprint_neighbourhood_suppresses_lower_peak(self):
        image = np.zeros((20, 20))
        image[10, 10] = 10.0
        image[10, 13] = 5.0   # inside the 7x7 footprint of the higher peak
        coords = peak_local_max(image, footprint=np.ones((7, 7), bool),
                                min_distance=1, exclude_border=False)
        assert len(coords) == 1
        assert tuple(coords[0]) == (10, 10)
 
    def test_footprint_keeps_well_separated_peaks(self):
        image = np.zeros((20, 20))
        image[5, 5] = 10.0
        image[15, 15] = 10.0
        coords = peak_local_max(image, footprint=np.ones((3, 3), bool),
                                min_distance=1, exclude_border=False)
        assert len(coords) == 2
        
class TestRemainingSignatureDefaults:
 
    def test_num_peaks_defaults(self):
        p = inspect.signature(peak_local_max).parameters
        assert p['num_peaks'].default == np.inf
        assert p['num_peaks_per_label'].default == np.inf
        
        
class TestWideRegionYieldsMultiplePeaks:
 
    def test_long_plateau_yields_multiple_spaced_peaks(self):
        """An 81-px tied plateau with min_distance=10 spans 8x min_distance,
        so several mutually-spaced representatives exist and must be
        returned. An implementation that collapses each connected tied
        region to a single representative returns only 1."""
        image = np.zeros((30, 200))
        image[15, 10:91] = 5.0
        coords = peak_local_max(image, min_distance=10, exclude_border=False)
        assert len(coords) >= 2
        _assert_min_distance(coords, 10)
 
    def test_long_plateau_within_label_yields_multiple(self):
        """Same requirement inside a labeled region."""
        image = np.zeros((30, 200))
        labels = np.zeros((30, 200), dtype=int)
        labels[10:21, :] = 1
        image[15, 10:91] = 5.0
        coords = peak_local_max(image, labels=labels, min_distance=10,
                                exclude_border=False)
        assert len(coords) >= 2
        _assert_min_distance(coords, 10)
 
    def test_large_block_plateau_yields_multiple(self):
        """A 2-D tied block far larger than min_distance."""
        image = np.zeros((60, 60))
        image[10:51, 10:51] = 5.0
        coords = peak_local_max(image, min_distance=10, exclude_border=False)
        assert len(coords) >= 4
        _assert_min_distance(coords, 10)
 