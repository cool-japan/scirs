"""
SciPy Comparison Tests for Spatial Module

Compares scirs2 spatial/distance functions against SciPy.spatial
to ensure numerical correctness and API compatibility.
"""

import numpy as np
import pytest
import scipy.spatial.distance
import scirs2


class TestDistanceFunctions:
    """Test pairwise distance functions"""

    def test_euclidean_distance(self):
        """Euclidean distance should match SciPy"""
        np.random.seed(42)

        for dim in [2, 5, 10]:
            u = np.ascontiguousarray(np.random.randn(dim))
            v = np.ascontiguousarray(np.random.randn(dim))

            # SciPy Euclidean
            scipy_dist = scipy.spatial.distance.euclidean(u, v)

            # SciRS2 Euclidean
            scirs2_dist = scirs2.euclidean_py(u, v)

            assert np.allclose(scipy_dist, scirs2_dist, rtol=1e-12)

    def test_cosine_distance(self):
        """Cosine distance should match SciPy"""
        np.random.seed(42)

        for dim in [2, 5, 10]:
            u = np.ascontiguousarray(np.random.randn(dim))
            v = np.ascontiguousarray(np.random.randn(dim))

            # SciPy cosine distance
            scipy_dist = scipy.spatial.distance.cosine(u, v)

            # SciRS2 cosine distance
            scirs2_dist = scirs2.cosine_py(u, v)

            assert np.allclose(scipy_dist, scirs2_dist, rtol=1e-12)

    def test_hamming_distance(self):
        """Hamming distance should match SciPy

        NOTE: scirs2.hamming_py is actually a Hamming window function for signal
        processing, not a distance metric -- use scirs2.spatial_hamming_distance_py
        instead, which is the SciPy-compatible Hamming distance metric. The name
        hamming_distance_py is intentionally avoided as well, since it is reserved
        for the text module's string/character Hamming distance.
        """
        np.random.seed(42)

        # Binary vectors (create as int then convert to float)
        u = np.ascontiguousarray(np.random.randint(0, 2, 20).astype(float))
        v = np.ascontiguousarray(np.random.randint(0, 2, 20).astype(float))

        # SciPy Hamming
        scipy_dist = scipy.spatial.distance.hamming(u, v)

        # SciRS2 Hamming
        scirs2_dist = scirs2.spatial_hamming_distance_py(u, v)

        assert np.allclose(scipy_dist, scirs2_dist, rtol=1e-12)


class TestPairwiseDistances:
    """Test pairwise distance computations"""

    def test_pdist_euclidean(self):
        """Pairwise distances should match SciPy"""
        np.random.seed(42)

        X = np.ascontiguousarray(np.random.randn(10, 3))

        # SciPy pdist
        scipy_dists = scipy.spatial.distance.pdist(X, metric='euclidean')

        # SciRS2 pdist
        scirs2_dists = scirs2.pdist_py(X, metric='euclidean')

        assert np.allclose(scipy_dists, scirs2_dists, rtol=1e-10)

    def test_cdist_euclidean(self):
        """Cross-distances should match SciPy"""
        np.random.seed(42)

        X = np.ascontiguousarray(np.random.randn(10, 3))
        Y = np.ascontiguousarray(np.random.randn(15, 3))

        # SciPy cdist
        scipy_dists = scipy.spatial.distance.cdist(X, Y, metric='euclidean')

        # SciRS2 cdist
        scirs2_dists = scirs2.cdist_py(X, Y, metric='euclidean')

        assert scipy_dists.shape == scirs2_dists.shape
        assert np.allclose(scipy_dists, scirs2_dists, rtol=1e-10)


class TestKDTree:
    """Test KD-tree spatial data structure"""

    def test_kdtree_query(self):
        """KDTree nearest neighbor queries should work"""
        np.random.seed(42)

        # Build tree
        data = np.ascontiguousarray(np.random.randn(100, 3))
        tree = scirs2.KDTree(data)

        # Query point
        query_point = np.ascontiguousarray(np.array([0.0, 0.0, 0.0]))

        # Find nearest neighbor (k=1, returns dict)
        result = tree.query(query_point, k=1)
        indices = result['indices']
        distances = result['distances']

        # Extract single values
        index = indices[0]
        distance = distances[0]

        # Verify result is reasonable
        assert 0 <= index < len(data)
        assert distance >= 0

        # Verify it's actually the nearest
        manual_distances = np.sqrt(np.sum((data - query_point)**2, axis=1))
        min_dist = manual_distances.min()
        assert np.allclose(distance, min_dist, rtol=1e-10)

    def test_kdtree_query_multiple(self):
        """KDTree k-nearest neighbors should work"""
        np.random.seed(42)

        data = np.ascontiguousarray(np.random.randn(50, 2))
        tree = scirs2.KDTree(data)

        query_point = np.ascontiguousarray(np.array([0.0, 0.0]))

        # Find 5 nearest neighbors (returns dict)
        result = tree.query(query_point, k=5)
        indices = result['indices']
        distances = result['distances']

        # Should return 5 results
        assert len(distances) == 5
        assert len(indices) == 5

        # Distances should be in ascending order
        assert np.all(distances[:-1] <= distances[1:])

        # All indices should be valid
        assert np.all(indices >= 0)
        assert np.all(indices < len(data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
