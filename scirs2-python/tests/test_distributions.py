"""Tests for statistical distributions."""

import pytest
import numpy as np
import scirs2


class TestNormalDistribution:
    """Test Normal (Gaussian) distribution."""

    def test_norm_creation(self):
        """Test creating normal distribution."""
        dist = scirs2.norm()
        assert dist is not None

        dist_custom = scirs2.norm(loc=5.0, scale=2.0)
        assert dist_custom is not None

    def test_norm_pdf(self):
        """Test normal PDF."""
        dist = scirs2.norm()  # Standard normal

        # PDF at 0 for standard normal should be ~0.3989
        pdf_at_zero = dist.pdf(0.0)
        assert abs(pdf_at_zero - 0.3989423) < 1e-5

        # PDF should be symmetric
        assert abs(dist.pdf(1.0) - dist.pdf(-1.0)) < 1e-10

    def test_norm_cdf(self):
        """Test normal CDF."""
        dist = scirs2.norm()

        # CDF at 0 for standard normal should be 0.5
        cdf_at_zero = dist.cdf(0.0)
        assert abs(cdf_at_zero - 0.5) < 1e-10

        # CDF at infinity should approach 1
        assert dist.cdf(10.0) > 0.999

    def test_norm_ppf(self):
        """Test normal PPF (inverse CDF)."""
        dist = scirs2.norm()

        # PPF at 0.5 should be close to 0
        median = dist.ppf(0.5)
        assert abs(median) < 1e-5

        # PPF at 0.9772 should be ~2 (approximately)
        val = dist.ppf(0.9772)
        assert abs(val - 2.0) < 0.05

    def test_norm_rvs(self):
        """Test normal random variates."""
        dist = scirs2.norm()

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # Mean should be close to 0
        mean = np.mean(samples)
        assert abs(mean) < 0.15

        # Std should be close to 1
        std = np.std(samples, ddof=1)
        assert abs(std - 1.0) < 0.15


class TestBinomialDistribution:
    """Test Binomial distribution."""

    def test_binom_creation(self):
        """Test creating binomial distribution."""
        dist = scirs2.binom(10, 0.5)
        assert dist is not None

    def test_binom_pmf(self):
        """Test binomial PMF."""
        dist = scirs2.binom(10, 0.5)

        # PMF at k=5 for n=10, p=0.5 should be highest
        pmf_5 = dist.pmf(5.0)
        pmf_0 = dist.pmf(0.0)
        pmf_10 = dist.pmf(10.0)

        assert pmf_5 > pmf_0
        assert pmf_5 > pmf_10

    def test_binom_cdf(self):
        """Test binomial CDF."""
        dist = scirs2.binom(10, 0.5)

        # CDF should be monotonically increasing
        cdf_0 = dist.cdf(0.0)
        cdf_5 = dist.cdf(5.0)
        cdf_10 = dist.cdf(10.0)

        assert cdf_0 < cdf_5 < cdf_10
        assert abs(cdf_10 - 1.0) < 1e-10

    def test_binom_ppf(self):
        """Test binomial PPF."""
        dist = scirs2.binom(10, 0.5)

        # PPF at 0.5 should be around the median (5)
        median = dist.ppf(0.5)
        assert 4.0 <= median <= 6.0

    def test_binom_rvs(self):
        """Test binomial random variates."""
        dist = scirs2.binom(10, 0.5)

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # All samples should be between 0 and 10
        assert all(0 <= s <= 10 for s in samples)

        # Mean should be close to n*p = 5
        mean = np.mean(samples)
        assert abs(mean - 5.0) < 0.5


class TestPoissonDistribution:
    """Test Poisson distribution."""

    def test_poisson_creation(self):
        """Test creating Poisson distribution."""
        dist = scirs2.poisson(3.0)
        assert dist is not None

    def test_poisson_pmf(self):
        """Test Poisson PMF."""
        dist = scirs2.poisson(3.0)

        # PMF should be positive
        pmf_3 = dist.pmf(3.0)
        assert pmf_3 > 0

        # PMF at mean should be near maximum
        pmf_0 = dist.pmf(0.0)
        assert pmf_3 > pmf_0

    def test_poisson_cdf(self):
        """Test Poisson CDF."""
        dist = scirs2.poisson(3.0)

        # CDF should be monotonically increasing
        assert dist.cdf(0.0) < dist.cdf(3.0) < dist.cdf(10.0)

        # CDF at large k should approach 1
        assert dist.cdf(20.0) > 0.999

    def test_poisson_ppf(self):
        """Test Poisson PPF."""
        dist = scirs2.poisson(3.0)

        # PPF at 0.5 should be around the median (allowing wider range)
        median = dist.ppf(0.5)
        assert 1.0 <= median <= 5.0

    def test_poisson_rvs(self):
        """Test Poisson random variates."""
        dist = scirs2.poisson(3.0)

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # All samples should be non-negative
        assert all(s >= 0 for s in samples)

        # Mean should be close to lambda = 3.0
        mean = np.mean(samples)
        assert abs(mean - 3.0) < 0.5


class TestExponentialDistribution:
    """Test Exponential distribution."""

    def test_expon_creation(self):
        """Test creating exponential distribution."""
        dist = scirs2.expon()
        assert dist is not None

        dist_custom = scirs2.expon(scale=2.0)
        assert dist_custom is not None

    def test_expon_pdf(self):
        """Test exponential PDF."""
        dist = scirs2.expon()  # scale=1.0

        # PDF at 0 should be 1/scale = 1.0
        pdf_at_zero = dist.pdf(0.0)
        assert abs(pdf_at_zero - 1.0) < 1e-10

        # PDF should decrease
        assert dist.pdf(0.0) > dist.pdf(1.0) > dist.pdf(2.0)

    def test_expon_cdf(self):
        """Test exponential CDF."""
        dist = scirs2.expon()

        # CDF at 0 should be 0
        assert abs(dist.cdf(0.0)) < 1e-10

        # CDF should be monotonically increasing
        assert dist.cdf(0.0) < dist.cdf(1.0) < dist.cdf(2.0)

    def test_expon_ppf(self):
        """Test exponential PPF."""
        dist = scirs2.expon()

        # PPF at 0 should be 0
        assert abs(dist.ppf(0.0)) < 1e-10

        # PPF should be increasing
        assert dist.ppf(0.1) < dist.ppf(0.5) < dist.ppf(0.9)

    def test_expon_rvs(self):
        """Test exponential random variates."""
        dist = scirs2.expon()

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # All samples should be non-negative
        assert all(s >= 0 for s in samples)

        # Mean should be close to scale = 1.0
        mean = np.mean(samples)
        assert abs(mean - 1.0) < 0.2


class TestUniformDistribution:
    """Test Uniform distribution."""

    def test_uniform_creation(self):
        """Test creating uniform distribution."""
        dist = scirs2.uniform()
        assert dist is not None

        dist_custom = scirs2.uniform(loc=1.0, scale=3.0)
        assert dist_custom is not None

    def test_uniform_pdf(self):
        """Test uniform PDF."""
        dist = scirs2.uniform()  # [0, 1)

        # PDF should be constant = 1.0 in [0, 1)
        assert abs(dist.pdf(0.0) - 1.0) < 1e-10
        assert abs(dist.pdf(0.5) - 1.0) < 1e-10
        # PDF at boundary may be 0 (half-open interval)
        assert dist.pdf(0.99) > 0

    def test_uniform_cdf(self):
        """Test uniform CDF."""
        dist = scirs2.uniform()

        # CDF should be linear in [0, 1]
        assert abs(dist.cdf(0.0)) < 1e-10
        assert abs(dist.cdf(0.5) - 0.5) < 1e-10
        assert abs(dist.cdf(1.0) - 1.0) < 1e-10

    def test_uniform_ppf(self):
        """Test uniform PPF."""
        dist = scirs2.uniform()

        # PPF should be identity function in [0, 1]
        assert abs(dist.ppf(0.0)) < 1e-10
        assert abs(dist.ppf(0.5) - 0.5) < 1e-10
        assert abs(dist.ppf(1.0) - 1.0) < 1e-10

    def test_uniform_rvs(self):
        """Test uniform random variates."""
        dist = scirs2.uniform()

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # All samples should be in [0, 1]
        assert all(0 <= s <= 1 for s in samples)

        # Mean should be close to 0.5
        mean = np.mean(samples)
        assert abs(mean - 0.5) < 0.1


class TestDistributionConsistency:
    """Test consistency across distributions."""

    def test_cdf_ppf_inverse(self):
        """Test that PPF is the inverse of CDF."""
        dist = scirs2.norm()

        x = 1.5
        cdf_val = dist.cdf(x)
        ppf_val = dist.ppf(cdf_val)

        # Allow tolerance for numerical precision
        assert abs(ppf_val - x) < 1e-3

    def test_pdf_cdf_relationship(self):
        """Test relationship between PDF and CDF."""
        dist = scirs2.norm()

        # CDF should be monotonically increasing where PDF > 0
        x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
        cdf_vals = [dist.cdf(x) for x in x_vals]

        for i in range(len(cdf_vals) - 1):
            assert cdf_vals[i] < cdf_vals[i+1]
