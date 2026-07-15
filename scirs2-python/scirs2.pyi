"""Type stubs for scirs2 - Scientific Computing in Rust for Python."""

from typing import Optional, Dict, List, Any
import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Clustering Module
# =============================================================================

class KMeans:
    """K-Means clustering algorithm."""

    labels: NDArray[np.int32]
    inertia_: float

    def __init__(self, n_clusters: int) -> None: ...
    def fit(self, data: NDArray[np.float64]) -> None: ...

def silhouette_score_py(
    data: NDArray[np.float64],
    labels: NDArray[np.int32]
) -> float:
    """Calculate silhouette score for clustering evaluation."""
    ...

def davies_bouldin_score_py(
    data: NDArray[np.float64],
    labels: NDArray[np.int32]
) -> float:
    """Calculate Davies-Bouldin score for clustering evaluation."""
    ...

def calinski_harabasz_score_py(
    data: NDArray[np.float64],
    labels: NDArray[np.int32]
) -> float:
    """Calculate Calinski-Harabasz score for clustering evaluation."""
    ...

def standardize_py(
    data: NDArray[np.float64],
    with_mean: bool
) -> NDArray[np.float64]:
    """Standardize data (zero mean, unit variance)."""
    ...

def normalize_py(
    data: NDArray[np.float64],
    norm: str
) -> NDArray[np.float64]:
    """Normalize data rows (l1, l2, or max norm)."""
    ...

# =============================================================================
# Time Series Module
# =============================================================================

class PyTimeSeries:
    """Time series data container."""

    def __init__(
        self,
        values: NDArray[np.float64],
        timestamps: Optional[NDArray[np.float64]]
    ) -> None: ...

    def __len__(self) -> int: ...
    def set_frequency(self, frequency: float) -> None: ...
    def get_values(self) -> NDArray[np.float64]: ...
    def get_timestamps(self) -> Optional[NDArray[np.float64]]: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def describe(self) -> Dict[str, float]: ...

    @classmethod
    def from_pandas(cls, series: Any) -> "PyTimeSeries": ...

class PyARIMA:
    """ARIMA time series model."""

    def __init__(self, p: int, d: int, q: int) -> None: ...
    def fit(self, data: PyTimeSeries) -> None: ...
    def forecast(self, steps: int) -> NDArray[np.float64]: ...
    def get_params(self) -> Dict[str, float]: ...
    def get_ar_coefficients(self) -> NDArray[np.float64]: ...
    def get_ma_coefficients(self) -> NDArray[np.float64]: ...
    def summary(self) -> str: ...

def apply_differencing(
    data: PyTimeSeries,
    periods: int
) -> NDArray[np.float64]:
    """Apply differencing to a time series."""
    ...

def apply_seasonal_differencing(
    data: PyTimeSeries,
    periods: int
) -> NDArray[np.float64]:
    """Apply seasonal differencing to a time series."""
    ...

def stl_decomposition(
    data: PyTimeSeries,
    period: int
) -> Dict[str, NDArray[np.float64]]:
    """Perform STL decomposition. Returns dict with trend, seasonal, residual."""
    ...

def adf_test(
    data: PyTimeSeries,
    max_lags: Optional[int]
) -> Dict[str, float]:
    """Augmented Dickey-Fuller test for stationarity."""
    ...

def boxcox_transform(
    data: PyTimeSeries,
    lambda_param: Optional[float]
) -> Dict[str, Any]:
    """Box-Cox transformation. Returns dict with transformed data and lambda."""
    ...

def boxcox_inverse(
    data: NDArray[np.float64],
    lambda_param: float
) -> NDArray[np.float64]:
    """Inverse Box-Cox transformation."""
    ...

# =============================================================================
# Linear Algebra Module
# =============================================================================

def det_py(a: NDArray[np.float64]) -> float:
    """Compute matrix determinant."""
    ...

def inv_py(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute matrix inverse."""
    ...

def trace_py(a: NDArray[np.float64]) -> float:
    """Compute matrix trace."""
    ...

def lu_py(a: NDArray[np.float64]) -> Dict[str, NDArray[np.float64]]:
    """LU decomposition. Returns dict with L, U, P matrices."""
    ...

def qr_py(a: NDArray[np.float64]) -> Dict[str, NDArray[np.float64]]:
    """QR decomposition. Returns dict with Q, R matrices."""
    ...

def svd_py(
    a: NDArray[np.float64],
    full_matrices: bool = True
) -> Dict[str, NDArray[np.float64]]:
    """SVD decomposition. Returns dict with U, S, Vt matrices."""
    ...

def cholesky_py(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cholesky decomposition for positive definite matrices."""
    ...

def eig_py(a: NDArray[np.float64]) -> Dict[str, NDArray[np.float64]]:
    """Eigenvalue decomposition. Returns dict with eigenvalues and eigenvectors."""
    ...

def eigh_py(a: NDArray[np.float64]) -> Dict[str, NDArray[np.float64]]:
    """Eigenvalue decomposition for symmetric/Hermitian matrices."""
    ...

def eigvals_py(a: NDArray[np.float64]) -> Dict[str, NDArray[np.float64]]:
    """Compute eigenvalues only."""
    ...

def solve_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve linear system Ax = b."""
    ...

def lstsq_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64]
) -> Dict[str, Any]:
    """Least squares solution to Ax = b."""
    ...

def matrix_norm_py(
    a: NDArray[np.float64],
    ord: str = "fro"
) -> float:
    """Compute matrix norm (fro, 1, inf)."""
    ...

def vector_norm_py(
    x: NDArray[np.float64],
    ord: int = 2
) -> float:
    """Compute vector norm."""
    ...

def cond_py(a: NDArray[np.float64]) -> float:
    """Compute matrix condition number."""
    ...

def matrix_rank_py(
    a: NDArray[np.float64],
    tol: Optional[float] = None
) -> int:
    """Compute matrix rank."""
    ...

def pinv_py(
    a: NDArray[np.float64],
    rcond: Optional[float] = None
) -> NDArray[np.float64]:
    """
    Compute the Moore-Penrose pseudoinverse of a matrix.

    The pseudoinverse of a matrix A, denoted A⁺, is defined as the
    matrix that satisfies the Moore-Penrose conditions:
    - A A⁺ A = A
    - A⁺ A A⁺ = A⁺
    - (A A⁺)ᵀ = A A⁺
    - (A⁺ A)ᵀ = A⁺ A

    Parameters:
        a: Input matrix (m × n)
        rcond: Cutoff threshold for small singular values.
               Singular values smaller than rcond are treated as zero.
               If None, uses machine precision * max(m, n) * largest_singular_value

    Returns:
        Pseudoinverse of the matrix (n × m)

    Notes:
        - For invertible square matrices, pinv(A) = inv(A)
        - For rank-deficient matrices, provides the minimum-norm least-squares solution
        - Computed using SVD: A⁺ = V Σ⁺ Uᵀ where A = U Σ Vᵀ

    Examples:
        >>> a = np.array([[1, 2], [3, 4], [5, 6]])
        >>> a_pinv = scirs2.pinv_py(a)
        >>> # Verify A A⁺ A = A
        >>> np.allclose(a @ a_pinv @ a, a)
        True
    """
    ...

# =============================================================================
# Statistics Module
# =============================================================================

def describe_py(data: NDArray[np.float64]) -> Dict[str, float]:
    """Compute descriptive statistics (mean, std, var, min, max, median, count)."""
    ...

def mean_py(data: NDArray[np.float64]) -> float:
    """Compute mean."""
    ...

def std_py(data: NDArray[np.float64], ddof: int = 0) -> float:
    """Compute standard deviation."""
    ...

def var_py(data: NDArray[np.float64], ddof: int = 0) -> float:
    """Compute variance."""
    ...

def percentile_py(data: NDArray[np.float64], q: float) -> float:
    """Compute percentile (0-100)."""
    ...

def median_py(data: NDArray[np.float64]) -> float:
    """Compute median."""
    ...

def iqr_py(data: NDArray[np.float64]) -> float:
    """Compute interquartile range."""
    ...

def correlation_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64]
) -> float:
    """Compute Pearson correlation coefficient."""
    ...

def covariance_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    ddof: int = 1
) -> float:
    """Compute covariance."""
    ...

# Statistical tests
def ttest_1samp_py(
    data: NDArray[np.float64],
    popmean: float,
    alternative: str = "two-sided"
) -> Dict[str, float]:
    """
    One-sample t-test.

    Parameters:
        data: Sample data
        popmean: Population mean for null hypothesis
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dict with 'statistic', 'pvalue', 'df'
    """
    ...

def ttest_ind_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    equal_var: bool = True,
    alternative: str = "two-sided"
) -> Dict[str, float]:
    """
    Two-sample independent t-test.

    Parameters:
        a: First sample
        b: Second sample
        equal_var: If True, assume equal variance
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dict with 'statistic', 'pvalue', 'df'
    """
    ...

def ttest_rel_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    alternative: str = "two-sided"
) -> Dict[str, float]:
    """
    Paired (related samples) t-test.

    Tests whether the means of two related/paired samples differ. This is a
    parametric test for paired observations (e.g., before/after measurements,
    matched pairs, repeated measures on the same subjects).

    The paired t-test works by computing the differences between paired
    observations (d_i = a_i - b_i), then performing a one-sample t-test on
    these differences to test if the mean difference is significantly different
    from zero.

    Parameters:
        a: First array of observations (e.g., before treatment)
        b: Second array of observations (e.g., after treatment)
           Must be same length as a (paired observations)
        alternative: Alternative hypothesis (default: "two-sided")
                    - "two-sided" or "two_sided": Test if means differ (μ_d ≠ 0)
                    - "less": Test if a < b (μ_d < 0)
                    - "greater": Test if a > b (μ_d > 0)

    Returns:
        Dictionary with:
        - statistic: t-statistic computed from the differences
        - pvalue: P-value for the hypothesis test
        - df: Degrees of freedom (n - 1, where n is number of pairs)

    Examples:
        >>> # Before and after treatment measurements
        >>> before = np.array([140, 145, 138, 142, 147], dtype=np.float64)
        >>> after = np.array([125, 130, 122, 128, 132], dtype=np.float64)
        >>> result = scirs2.ttest_rel_py(before, after)
        >>> print(f"t={result['statistic']:.3f}, p={result['pvalue']:.4f}")

        >>> # Test if treatment reduced values (one-sided test)
        >>> result = scirs2.ttest_rel_py(before, after, alternative="greater")
        >>> if result['pvalue'] < 0.05:
        ...     print("Treatment significantly reduced values")

        >>> # Matched pairs design (e.g., twins)
        >>> twin1 = np.array([75, 82, 79, 85, 88], dtype=np.float64)
        >>> twin2 = np.array([78, 85, 82, 89, 92], dtype=np.float64)
        >>> result = scirs2.ttest_rel_py(twin1, twin2)

    Notes:
        - **Assumptions**:
          1. Paired observations (each a_i paired with corresponding b_i)
          2. Differences are approximately normally distributed
          3. Observations are independent
        - **Use cases**:
          - Before/after measurements on same subjects
          - Matched pairs experimental designs
          - Repeated measures on same subjects
          - Crossover study designs
        - **When NOT to use**:
          - For independent samples → use ttest_ind_py instead
          - For non-normal differences → consider wilcoxon_py (signed-rank test)
        - The test is more powerful than independent t-test when observations
          are truly paired, as it accounts for within-pair correlation
        - If all differences are identical (zero variance), the test statistic
          will be infinite and p-value will be 0

    Statistical Background:
        The test statistic is computed as:
            t = (mean(d) - 0) / (std(d) / sqrt(n))
        where d = a - b are the paired differences, and follows a t-distribution
        with n-1 degrees of freedom under the null hypothesis (mean difference = 0).

    See Also:
        - ttest_1samp_py: One-sample t-test
        - ttest_ind_py: Independent two-sample t-test
        - wilcoxon_py: Non-parametric alternative for paired samples
    """
    ...

def shapiro_py(data: NDArray[np.float64]) -> Dict[str, float]:
    """
    Shapiro-Wilk test for normality.

    Tests the null hypothesis that the data was drawn from a normal distribution.

    Parameters:
        data: Sample data

    Returns:
        Dictionary with:
        - statistic: W statistic
        - pvalue: Two-sided p-value

    Notes:
        - Sample size should be between 3 and 5000
        - Higher W values indicate more normality
        - Reject normality if pvalue < significance level (e.g., 0.05)
    """
    ...

def chisquare_py(
    observed: NDArray[np.float64],
    expected: Optional[NDArray[np.float64]] = None
) -> Dict[str, Any]:
    """
    Chi-square goodness-of-fit test.

    Tests whether observed frequencies differ from expected frequencies.

    Parameters:
        observed: Observed frequencies
        expected: Expected frequencies (optional, defaults to uniform distribution)

    Returns:
        Dictionary with:
        - statistic: Chi-square statistic
        - pvalue: Two-sided p-value
        - dof: Degrees of freedom

    Examples:
        >>> observed = np.array([10, 15, 12, 13])
        >>> result = scirs2.chisquare_py(observed)
    """
    ...

def chi2_independence_py(observed: NDArray[np.int64]) -> Dict[str, Any]:
    """
    Chi-square test for independence in contingency tables.

    Tests whether two categorical variables are independent using a contingency
    table (cross-tabulation) of observed frequencies. Under the null hypothesis,
    the row and column variables are independent.

    Parameters:
        observed: 2D array of observed frequencies (contingency table)
                  Must have at least 2 rows and 2 columns
                  Each cell represents the count for a combination of categories

    Returns:
        Dictionary with:
        - statistic: Chi-square test statistic
        - pvalue: P-value for the test
        - df: Degrees of freedom ((rows-1) × (columns-1))
        - expected: 2D array of expected frequencies under independence

    Raises:
        RuntimeError: If the table has fewer than 2 rows or 2 columns

    Examples:
        >>> # Test independence between treatment and outcome (2×2 table)
        >>> observed = np.array([
        ...     [10, 20],  # Treatment A: Success, Failure
        ...     [15, 25]   # Treatment B: Success, Failure
        ... ], dtype=np.int64)
        >>> result = scirs2.chi2_independence_py(observed)
        >>> print(f"χ²={result['statistic']:.3f}, p={result['pvalue']:.4f}")

        >>> # Test independence between education and income (3×3 table)
        >>> observed = np.array([
        ...     [20, 30, 10],  # High school: Low, Medium, High income
        ...     [15, 25, 20],  # Bachelor's: Low, Medium, High income
        ...     [25, 15, 40]   # Graduate: Low, Medium, High income
        ... ], dtype=np.int64)
        >>> result = scirs2.chi2_independence_py(observed)
        >>> if result['pvalue'] < 0.05:
        ...     print("Variables are dependent (reject independence)")
        ... else:
        ...     print("Variables appear independent")

    Notes:
        - This test assumes all expected frequencies are at least 5
        - For 2×2 tables with small samples, consider using chi2_yates_py
        - Expected frequencies are calculated as:
          E[i,j] = (row_i_sum × col_j_sum) / total
        - The test statistic is: χ² = Σ[(O - E)² / E]
        - Degrees of freedom: df = (rows - 1) × (columns - 1)
        - Large χ² values (small p-values) indicate dependence
        - The test does not indicate the direction or nature of the association
    """
    ...

def chi2_yates_py(observed: NDArray[np.int64]) -> Dict[str, Any]:
    """
    Chi-square test with Yates' continuity correction for 2×2 tables.

    Applies Yates' continuity correction to improve the chi-square approximation
    for 2×2 contingency tables, especially with small sample sizes. The correction
    reduces the chi-square statistic, making the test more conservative.

    Parameters:
        observed: 2×2 array of observed frequencies
                  Must be exactly 2 rows and 2 columns

    Returns:
        Dictionary with:
        - statistic: Chi-square test statistic with Yates' correction
        - pvalue: P-value for the test
        - df: Degrees of freedom (always 1 for 2×2 tables)
        - expected: 2×2 array of expected frequencies under independence

    Raises:
        RuntimeError: If the table is not 2×2

    Examples:
        >>> # Test with small sample (where Yates' correction is beneficial)
        >>> observed = np.array([
        ...     [8, 12],   # Group 1: Success, Failure
        ...     [10, 15]   # Group 2: Success, Failure
        ... ], dtype=np.int64)
        >>> result = scirs2.chi2_yates_py(observed)
        >>> print(f"χ²={result['statistic']:.3f}, p={result['pvalue']:.4f}")

        >>> # Compare with standard chi-square test
        >>> yates_result = scirs2.chi2_yates_py(observed)
        >>> chi2_result = scirs2.chi2_independence_py(observed)
        >>> print(f"Yates: χ²={yates_result['statistic']:.3f}")
        >>> print(f"Standard: χ²={chi2_result['statistic']:.3f}")
        >>> # Yates gives lower (more conservative) statistic

    Notes:
        - **Only for 2×2 tables** - use chi2_independence_py for larger tables
        - Recommended when expected frequencies are between 5 and 10
        - The correction subtracts 0.5 from |O - E| before squaring:
          χ²_Yates = Σ[(|O - E| - 0.5)² / E]
        - Makes the test more conservative (higher p-values)
        - More accurate p-values for small samples than uncorrected test
        - For very small samples (expected < 5), consider Fisher's exact test
        - Degrees of freedom is always 1 for 2×2 tables
    """
    ...

# Contingency table analysis
def fisher_exact_py(
    table: NDArray[np.float64],
    alternative: str = "two-sided"
) -> Dict[str, float]:
    """
    Fisher's exact test for 2×2 contingency tables.

    Performs Fisher's exact test, which calculates the exact probability of
    observing a table at least as extreme as the one given, under the null
    hypothesis of independence. Particularly useful for small sample sizes
    where chi-square approximation may not be valid.

    Fisher's exact test is based on the hypergeometric distribution and
    provides exact p-values without relying on large-sample approximations.

    Parameters:
        table: 2×2 array of observed frequencies
               Must be exactly 2 rows and 2 columns
               [[a, b],
                [c, d]]
               where a, b, c, d are counts (frequencies)
        alternative: Alternative hypothesis (default: "two-sided")
                    - "two-sided": Test if association exists (OR ≠ 1)
                    - "less": Test if odds ratio < 1
                    - "greater": Test if odds ratio > 1

    Returns:
        Dictionary with:
        - odds_ratio: Odds ratio (a*d)/(b*c)
                     Measure of association strength
        - pvalue: Exact p-value for the test

    Examples:
        >>> # Example: Treatment effectiveness study
        >>> # Rows: Treatment (Yes/No), Cols: Improved (Yes/No)
        >>> table = np.array([[8, 2],   # Treated: 8 improved, 2 not
        ...                   [1, 5]],  # Control: 1 improved, 5 not
        ...                  dtype=np.float64)
        >>> result = scirs2.fisher_exact_py(table)
        >>> print(f"OR={result['odds_ratio']:.2f}, p={result['pvalue']:.4f}")
        >>> if result['pvalue'] < 0.05:
        ...     print("Significant association found")

        >>> # Example: Small sample case (where chi-square wouldn't be valid)
        >>> table = np.array([[2, 1], [1, 2]], dtype=np.float64)
        >>> result = scirs2.fisher_exact_py(table)

        >>> # One-sided test for increased odds
        >>> result = scirs2.fisher_exact_py(table, alternative="greater")

    Notes:
        - **Use when**:
          * Sample size is small (expected frequencies < 5)
          * Need exact p-values (not approximations)
          * 2×2 contingency table analysis
        - **Assumptions**:
          * Fixed row and column margins (hypergeometric model)
          * Independent observations
          * All values must be non-negative counts
        - **Interpretation**:
          * Odds ratio = 1: No association
          * Odds ratio > 1: Positive association (rows/cols concordant)
          * Odds ratio < 1: Negative association (rows/cols discordant)
          * Odds ratio = 0 or ∞: Perfect association
        - **Comparison with chi-square**:
          * Fisher's exact: Always valid, computationally intensive for large tables
          * Chi-square: Approximation, requires expected frequencies ≥ 5
          * For large samples, both give similar results
        - The test is "exact" because it computes the probability directly
          using the hypergeometric distribution, not a large-sample approximation

    Statistical Background:
        Under the null hypothesis of independence, the probability of observing
        the table [[a,b],[c,d]] with fixed margins is:
            P = [(a+b)!(c+d)!(a+c)!(b+d)!] / [n!a!b!c!d!]
        where n = a+b+c+d is the total sample size.

    See Also:
        - chi2_independence_py: Chi-square test for larger tables
        - chi2_yates_py: Chi-square with Yates' correction for 2×2 tables
        - odds_ratio_py: Calculate odds ratio only
    """
    ...

def odds_ratio_py(table: NDArray[np.float64]) -> float:
    """
    Calculate odds ratio for a 2×2 contingency table.

    The odds ratio (OR) is a measure of association between an exposure and
    an outcome. It quantifies how strongly the presence of exposure is
    associated with the presence of outcome.

    For a 2×2 contingency table:
                  Outcome+  Outcome-
       Exposed+      a         b
       Exposed-      c         d

    Odds Ratio = (a × d) / (b × c)

    Parameters:
        table: 2×2 array of observed frequencies
               Must be exactly 2 rows and 2 columns

    Returns:
        Odds ratio value (float)

    Examples:
        >>> # Example: Case-control study
        >>> # Rows: Cases/Controls, Cols: Exposed/Not Exposed
        >>> table = np.array([[10, 5],   # Cases: 10 exposed, 5 not
        ...                   [3, 12]],  # Controls: 3 exposed, 12 not
        ...                  dtype=np.float64)
        >>> or_val = scirs2.odds_ratio_py(table)
        >>> print(f"Odds Ratio: {or_val:.2f}")
        >>> # OR = (10*12)/(5*3) = 120/15 = 8.0
        >>> # Exposed individuals have 8× the odds of being cases

        >>> # Example: Equal odds (no association)
        >>> table = np.array([[10, 10], [10, 10]], dtype=np.float64)
        >>> or_val = scirs2.odds_ratio_py(table)
        >>> print(f"OR: {or_val:.2f}")  # OR = 1.0

        >>> # Example: Protective factor (OR < 1)
        >>> table = np.array([[2, 10], [10, 5]], dtype=np.float64)
        >>> or_val = scirs2.odds_ratio_py(table)
        >>> print(f"OR: {or_val:.2f}")  # OR = 0.1

    Notes:
        - **Interpretation**:
          * OR = 1: No association (exposure doesn't affect odds)
          * OR > 1: Positive association (exposure increases odds)
          * OR < 1: Negative association (exposure decreases odds / protective)
          * OR = 0: Impossible outcome when exposed
          * OR = ∞: Impossible outcome when not exposed
        - **Use cases**:
          * Case-control studies (preferred over relative risk)
          * Cross-sectional studies
          * Logistic regression interpretation
        - **Relationship to relative risk**:
          * For rare outcomes (< 10% incidence): OR ≈ RR
          * For common outcomes: OR > RR (overestimates risk)
          * OR is always further from 1 than RR
        - **Confidence intervals**:
          * Use Fisher's exact test to get p-values
          * 95% CI can be computed as: exp(ln(OR) ± 1.96 × SE)
          * where SE = sqrt(1/a + 1/b + 1/c + 1/d)
        - **Continuity correction**:
          * For zero cells, add 0.5 to all cells: (a+0.5)(d+0.5)/(b+0.5)(c+0.5)
          * This implementation uses exact calculation

    Statistical Background:
        The odds ratio compares the odds of outcome in the exposed group
        (a/b) to the odds in the unexposed group (c/d):
            OR = (a/b) / (c/d) = (a×d) / (b×c)

    See Also:
        - fisher_exact_py: Includes OR with significance test
        - relative_risk_py: Alternative measure for cohort studies
    """
    ...

def relative_risk_py(table: NDArray[np.float64]) -> float:
    """
    Calculate relative risk (risk ratio) for a 2×2 contingency table.

    The relative risk (RR) or risk ratio is a measure of association between
    an exposure and an outcome in cohort studies. It compares the risk of
    outcome in the exposed group to the risk in the unexposed group.

    For a 2×2 contingency table:
                  Outcome+  Outcome-
       Exposed+      a         b
       Exposed-      c         d

    Relative Risk = [a/(a+b)] / [c/(c+d)]

    Parameters:
        table: 2×2 array of observed frequencies
               Must be exactly 2 rows and 2 columns

    Returns:
        Relative risk value (float)

    Examples:
        >>> # Example: Cohort study
        >>> # Rows: Exposed/Unexposed, Cols: Disease+/Disease-
        >>> table = np.array([[20, 80],   # Exposed: 20 diseased, 80 not
        ...                   [10, 90]],  # Unexposed: 10 diseased, 90 not
        ...                  dtype=np.float64)
        >>> rr_val = scirs2.relative_risk_py(table)
        >>> print(f"Relative Risk: {rr_val:.2f}")
        >>> # RR = (20/100)/(10/100) = 0.2/0.1 = 2.0
        >>> # Exposed group has 2× the risk of disease

        >>> # Example: No association (RR = 1)
        >>> table = np.array([[10, 40], [10, 40]], dtype=np.float64)
        >>> rr_val = scirs2.relative_risk_py(table)
        >>> print(f"RR: {rr_val:.2f}")  # RR = 1.0

        >>> # Example: Protective factor (RR < 1)
        >>> table = np.array([[5, 45], [20, 30]], dtype=np.float64)
        >>> rr_val = scirs2.relative_risk_py(table)
        >>> print(f"RR: {rr_val:.2f}")  # RR = 0.25

    Notes:
        - **Interpretation**:
          * RR = 1: No association (equal risk in both groups)
          * RR > 1: Positive association (exposure increases risk)
          * RR < 1: Negative association (exposure decreases risk / protective)
          * RR = 0: No cases in exposed group
          * RR = ∞: No cases in unexposed group
        - **Use cases**:
          * Cohort studies (prospective or retrospective)
          * Randomized controlled trials
          * When disease incidence can be directly measured
        - **Comparison with odds ratio**:
          * For rare diseases (< 10% incidence): RR ≈ OR
          * For common diseases: RR < OR
          * RR is always closer to 1 than OR
          * RR is more intuitive and easier to interpret
        - **When NOT to use**:
          * Case-control studies (can't estimate incidence)
          * Use odds ratio instead for case-control designs
        - **Confidence intervals**:
          * 95% CI can be computed as: exp(ln(RR) ± 1.96 × SE)
          * where SE = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
        - **Attributable risk**:
          * AR = [a/(a+b)] - [c/(c+d)] = incidence difference
          * AR% = [(RR-1)/RR] × 100 = attributable risk percent

    Statistical Background:
        The relative risk compares the probability (risk) of outcome in the
        exposed group to the probability in the unexposed group:
            RR = P(Outcome|Exposed) / P(Outcome|Unexposed)
               = [a/(a+b)] / [c/(c+d)]

    See Also:
        - odds_ratio_py: Alternative measure for case-control studies
        - fisher_exact_py: Significance test with odds ratio
    """
    ...

# Linear regression
def linregress_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64]
) -> Dict[str, float]:
    """
    Calculate a simple linear regression.

    Performs ordinary least squares (OLS) linear regression to fit a line
    y = slope × x + intercept to the data. Computes the correlation coefficient,
    p-value for testing H₀: slope = 0, and standard error of the slope estimate.

    This function uses the method of least squares to estimate the parameters
    of a linear model. It minimizes the sum of squared residuals between the
    observed values and the values predicted by the linear model.

    Parameters:
        x: Independent variable (predictor), 1D array
        y: Dependent variable (response), 1D array
           Must be same length as x

    Returns:
        Dictionary with:
        - slope: Slope of the regression line (b₁)
                Interpretation: change in y per unit change in x
        - intercept: Y-intercept of the regression line (b₀)
                    Interpretation: predicted y when x = 0
        - rvalue: Pearson correlation coefficient (r)
                 Ranges from -1 to 1, measures linear association strength
        - pvalue: Two-sided p-value for testing H₀: slope = 0
                 Tests whether the slope is significantly different from zero
        - stderr: Standard error of the slope estimate
                 Measures uncertainty in the slope estimate

    Examples:
        >>> # Example: Perfect linear relationship
        >>> x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        >>> y = np.array([2, 4, 6, 8, 10], dtype=np.float64)  # y = 2x
        >>> result = scirs2.linregress_py(x, y)
        >>> print(f"y = {result['slope']:.2f}x + {result['intercept']:.2f}")
        >>> print(f"R = {result['rvalue']:.4f}, p = {result['pvalue']:.6f}")

        >>> # Example: Temperature conversion (Celsius to Fahrenheit)
        >>> celsius = np.array([0, 10, 20, 30, 40], dtype=np.float64)
        >>> fahrenheit = np.array([32, 50, 68, 86, 104], dtype=np.float64)
        >>> result = scirs2.linregress_py(celsius, fahrenheit)
        >>> # F = 1.8C + 32
        >>> print(f"Conversion: F = {result['slope']:.1f}C + {result['intercept']:.1f}")

        >>> # Example: Prediction
        >>> x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        >>> y = np.array([2.1, 3.9, 6.1, 7.9, 10.2], dtype=np.float64)
        >>> result = scirs2.linregress_py(x, y)
        >>> # Predict y for new x value
        >>> x_new = 6.0
        >>> y_pred = result['slope'] * x_new + result['intercept']
        >>> print(f"Predicted y at x={x_new}: {y_pred:.2f}")

        >>> # Example: Check significance
        >>> if result['pvalue'] < 0.05:
        ...     print("Relationship is statistically significant")
        ...     print(f"95% CI for slope: {result['slope']} ± {1.96 * result['stderr']:.4f}")

    Notes:
        - **Model**: y = slope × x + intercept + ε, where ε ~ N(0, σ²)
        - **Assumptions**:
          * Linearity: True relationship between x and y is linear
          * Independence: Observations are independent
          * Homoscedasticity: Constant variance of residuals
          * Normality: Residuals are approximately normally distributed (for inference)
        - **Interpretation**:
          * slope: Amount y changes for one-unit increase in x
          * intercept: Expected value of y when x = 0
          * r²: Proportion of variance in y explained by x (r² = rvalue²)
          * pvalue: Probability of observing data if true slope were zero
          * stderr: Uncertainty in slope estimate (use for confidence intervals)
        - **When to use**:
          * Simple bivariate relationship
          * Quick exploratory analysis
          * Single predictor variable
        - **Limitations**:
          * Only for linear relationships (use nonlinear regression for curves)
          * Sensitive to outliers (consider robust regression)
          * Only one predictor (use multiple regression for >1 predictor)
          * Assumes fixed x, random y (use ODR if both have error)
        - **Perfect correlation** (|r| = 1):
          * pvalue may be NaN due to division by zero in t-statistic
          * stderr = 0 (perfect fit)
          * This is expected behavior for exact linear relationships
        - **Goodness of fit**:
          * R² (coefficient of determination) = rvalue²
          * R² = 1: Perfect fit
          * R² = 0: No linear relationship
          * Closer to 1 indicates better fit

    Statistical Background:
        The slope is estimated as:
            slope = Cov(x, y) / Var(x) = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]

        The intercept is:
            intercept = ȳ - slope × x̄

        The correlation coefficient is:
            r = Cov(x, y) / [SD(x) × SD(y)]

        The p-value is computed from the t-statistic:
            t = r × √(n-2) / √(1 - r²)
        which follows a t-distribution with n-2 degrees of freedom.

    See Also:
        - pearsonr_py: Compute correlation without regression
        - For multiple predictors: Use dedicated regression libraries
        - For robust regression: Consider specialized robust methods
        - For nonlinear fits: Consider polynomial or nonlinear regression
    """
    ...

def f_oneway_py(*args: NDArray[np.float64]) -> Dict[str, float]:
    """
    One-way ANOVA (Analysis of Variance).

    Tests whether the means of multiple groups are equal.

    Parameters:
        *args: Two or more arrays, each representing a group

    Returns:
        Dictionary with:
        - f_statistic: F-statistic
        - pvalue: Two-sided p-value
        - df_between: Degrees of freedom between groups
        - df_within: Degrees of freedom within groups
        - ss_between: Sum of squares between groups
        - ss_within: Sum of squares within groups
        - ms_between: Mean square between groups
        - ms_within: Mean square within groups

    Examples:
        >>> group1 = np.array([85, 82, 78, 88, 91])
        >>> group2 = np.array([76, 80, 82, 84, 79])
        >>> group3 = np.array([91, 89, 93, 87, 90])
        >>> result = scirs2.f_oneway_py(group1, group2, group3)
    """
    ...

def tukey_hsd_py(*args: NDArray[np.float64], alpha: float = 0.05) -> List[Dict[str, Any]]:
    """
    Tukey's Honestly Significant Difference (HSD) post-hoc test.

    Performs pairwise comparisons between group means after ANOVA, controlling
    the family-wise error rate. This test identifies which specific groups differ
    from each other when ANOVA shows a significant overall effect.

    Parameters:
        *args: Two or more arrays, each representing a group
               All groups must have at least 2 observations
        alpha: Significance level (default: 0.05)
               Supported values: 0.05 or 0.01

    Returns:
        List of dictionaries, one for each pairwise comparison:
        - group1: Index of first group (0-based)
        - group2: Index of second group (0-based)
        - mean_diff: Difference between group means
        - pvalue: P-value for the comparison
        - significant: Boolean indicating if difference is significant at alpha level

    Raises:
        RuntimeError: If fewer than 2 groups provided, or if alpha is not 0.05 or 0.01

    Examples:
        >>> # Post-hoc test after finding significant ANOVA
        >>> group1 = np.array([85.0, 82.0, 78.0, 88.0, 91.0])
        >>> group2 = np.array([76.0, 80.0, 82.0, 84.0, 79.0])
        >>> group3 = np.array([91.0, 89.0, 93.0, 87.0, 90.0])
        >>>
        >>> # First, perform ANOVA
        >>> anova_result = scirs2.f_oneway_py(group1, group2, group3)
        >>> if anova_result['pvalue'] < 0.05:
        ...     # Significant ANOVA, perform post-hoc tests
        ...     comparisons = scirs2.tukey_hsd_py(group1, group2, group3)
        ...     for comp in comparisons:
        ...         print(f"Group {comp['group1']} vs {comp['group2']}: "
        ...               f"diff={comp['mean_diff']:.2f}, p={comp['pvalue']:.4f}, "
        ...               f"sig={comp['significant']}")

        >>> # Use custom alpha level
        >>> comparisons = scirs2.tukey_hsd_py(group1, group2, group3, alpha=0.01)
        >>> # More stringent: fewer comparisons will be significant

        >>> # With 4 groups, get 6 pairwise comparisons: C(4,2) = 6
        >>> g1 = np.array([5.0, 6.0, 7.0])
        >>> g2 = np.array([8.0, 9.0, 10.0])
        >>> g3 = np.array([11.0, 12.0, 13.0])
        >>> g4 = np.array([14.0, 15.0, 16.0])
        >>> comparisons = scirs2.tukey_hsd_py(g1, g2, g3, g4)
        >>> print(f"Number of comparisons: {len(comparisons)}")  # 6

    Notes:
        - **Use only after significant ANOVA result** - Tukey HSD is a post-hoc test
        - Controls family-wise error rate at the specified alpha level
        - More conservative than multiple t-tests (avoids inflated Type I error)
        - Based on the studentized range distribution (Q distribution)
        - Assumes:
          * Independent observations
          * Normality within groups
          * Homogeneity of variance (equal variances across groups)
        - For k groups, performs k(k-1)/2 pairwise comparisons
        - All comparisons use pooled within-group variance from all groups
        - Alpha = 0.05 or 0.01 supported (uses critical value approximation)
        - Groups can have different sample sizes (unequal n)
        - If ANOVA is not significant, post-hoc tests may not be appropriate
        - Alternative post-hoc tests: Bonferroni, Scheffé, Dunnett
    """
    ...

# Correlation tests with significance
def pearsonr_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Pearson correlation coefficient with significance test.

    Calculates the Pearson product-moment correlation coefficient between two
    arrays and tests for statistical significance. The Pearson correlation measures
    linear association between variables.

    Parameters:
        x: First array of observations
        y: Second array of observations (must be same length as x)
        alternative: Type of hypothesis test (default: "two-sided")
                    - "two-sided": Test if correlation is nonzero
                    - "less": Test if correlation is negative
                    - "greater": Test if correlation is positive

    Returns:
        Dictionary with:
        - correlation: Pearson correlation coefficient (r) in [-1, 1]
        - pvalue: P-value for testing non-correlation

    Raises:
        RuntimeError: If arrays have different lengths, contain insufficient data,
                     or have zero variance

    Examples:
        >>> # Perfect positive linear correlation
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> result = scirs2.pearsonr_py(x, y)
        >>> print(f"r={result['correlation']:.3f}, p={result['pvalue']:.4f}")
        r=1.000, p=0.0000

        >>> # Moderate correlation
        >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9])
        >>> result = scirs2.pearsonr_py(x, y)
        >>> if result['pvalue'] < 0.05:
        ...     print(f"Significant correlation: r={result['correlation']:.3f}")

        >>> # One-sided test for positive correlation
        >>> result = scirs2.pearsonr_py(x, y, alternative="greater")

    Notes:
        - Pearson correlation assumes linear relationship between variables
        - Sensitive to outliers
        - Requires continuous or interval data
        - Test statistic follows t-distribution with n-2 degrees of freedom
        - For nonlinear monotonic relationships, consider Spearman correlation
        - Range: -1 (perfect negative) to +1 (perfect positive), 0 = no correlation
        - P-value tests null hypothesis that true correlation is zero
    """
    ...

def spearmanr_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Spearman rank correlation coefficient with significance test.

    Calculates the Spearman rank-order correlation coefficient (Spearman's rho)
    and tests for statistical significance. Spearman correlation is a nonparametric
    measure of monotonic association that works with ranked data.

    Parameters:
        x: First array of observations
        y: Second array of observations (must be same length as x)
        alternative: Type of hypothesis test (default: "two-sided")
                    - "two-sided": Test if correlation is nonzero
                    - "less": Test if correlation is negative
                    - "greater": Test if correlation is positive

    Returns:
        Dictionary with:
        - correlation: Spearman rank correlation coefficient (rho) in [-1, 1]
        - pvalue: P-value for testing non-correlation

    Raises:
        RuntimeError: If arrays have different lengths or contain insufficient data

    Examples:
        >>> # Perfect monotonic relationship (not necessarily linear)
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        >>> y = x ** 2  # Nonlinear but monotonic
        >>> result = scirs2.spearmanr_py(x, y)
        >>> print(f"rho={result['correlation']:.3f}, p={result['pvalue']:.4f}")

        >>> # With ties in data
        >>> x = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        >>> y = np.array([5.0, 6.0, 6.0, 7.0, 8.0])
        >>> result = scirs2.spearmanr_py(x, y)

        >>> # One-sided test
        >>> result = scirs2.spearmanr_py(x, y, alternative="greater")

    Notes:
        - Nonparametric test (does not assume normality)
        - Robust to outliers compared to Pearson
        - Works with ordinal (ranked) data
        - Detects monotonic relationships (not just linear)
        - Converts data to ranks before computing correlation
        - Handles tied ranks appropriately
        - For n > 10, uses normal approximation for p-value
        - Range: -1 (perfect negative monotonic) to +1 (perfect positive monotonic)
        - Use when relationship is monotonic but not necessarily linear
        - Less powerful than Pearson for truly linear relationships
    """
    ...

def kendalltau_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    method: str = "b",
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Kendall tau rank correlation coefficient with significance test.

    Calculates Kendall's tau correlation coefficient and tests for statistical
    significance. Kendall's tau is a nonparametric measure of ordinal association
    based on concordant and discordant pairs.

    Parameters:
        x: First array of observations
        y: Second array of observations (must be same length as x)
        method: Kendall tau variant (default: "b")
               - "b": tau-b, accounts for ties in both variables
               - "c": tau-c, more suitable for rectangular contingency tables
        alternative: Type of hypothesis test (default: "two-sided")
                    - "two-sided": Test if correlation is nonzero
                    - "less": Test if correlation is negative
                    - "greater": Test if correlation is positive

    Returns:
        Dictionary with:
        - correlation: Kendall tau correlation coefficient in [-1, 1]
        - pvalue: P-value for testing non-correlation

    Raises:
        RuntimeError: If arrays have different lengths, contain insufficient data,
                     or invalid method specified

    Examples:
        >>> # Perfect concordance
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> result = scirs2.kendalltau_py(x, y)
        >>> print(f"tau={result['correlation']:.3f}, p={result['pvalue']:.4f}")

        >>> # With ties, using tau-b
        >>> x = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        >>> y = np.array([5.0, 6.0, 6.0, 7.0, 8.0])
        >>> result = scirs2.kendalltau_py(x, y, method="b")

        >>> # Using tau-c for rectangular tables
        >>> result = scirs2.kendalltau_py(x, y, method="c")

        >>> # One-sided test
        >>> result = scirs2.kendalltau_py(x, y, alternative="greater")

    Notes:
        - Nonparametric test (does not assume normality)
        - Based on concordant and discordant pairs
        - More robust to outliers than Pearson
        - Tau-b: Accounts for ties in both variables (most common)
        - Tau-c: Better for m×n contingency tables where m ≠ n
        - Typically has smaller magnitude than Spearman's rho
        - For n ≥ 10, uses normal approximation for p-value
        - Range: -1 (perfect negative association) to +1 (perfect positive)
        - Concordant pair: (x[i] - x[j]) and (y[i] - y[j]) have same sign
        - Discordant pair: opposite signs
        - tau = (concordant - discordant) / total_pairs
        - Less sensitive to errors in rankings than Spearman
        - Use when sample size is small or data is ordinal
    """
    ...

# Nonparametric tests
def wilcoxon_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    zero_method: str = "wilcox",
    correction: bool = True,
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Tests whether two paired samples have different distributions using
    ranks instead of actual values. This is a nonparametric alternative
    to the paired t-test.

    Parameters:
        x: First array of observations
        y: Second array of observations (paired with x)
        zero_method: How to handle zero differences: "wilcox" (default) or "pratt"
        correction: Whether to apply continuity correction (default: True)

    Returns:
        Dictionary with:
        - statistic: Wilcoxon signed-rank statistic
        - pvalue: Two-sided p-value

    Raises:
        RuntimeError: If all differences are zero or other test failures

    Examples:
        >>> before = np.array([120, 135, 128, 140, 125])
        >>> after = np.array([115, 130, 125, 138, 120])
        >>> result = scirs2.wilcoxon_py(before, after)
        >>> print(f"W={result['statistic']:.2f}, p={result['pvalue']:.4f}")
    """
    ...

def mannwhitneyu_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alternative: str = "two-sided",
    use_continuity: bool = True,
) -> Dict[str, float]:
    """
    Mann-Whitney U test for independent samples.

    Tests whether two independent samples have different distributions using
    ranks. This is a nonparametric alternative to the independent t-test.

    Parameters:
        x: First array of observations
        y: Second array of observations
        alternative: Alternative hypothesis: "two-sided" (default), "less", or "greater"
        use_continuity: Whether to apply continuity correction (default: True)

    Returns:
        Dictionary with:
        - statistic: Mann-Whitney U statistic
        - pvalue: p-value for the test

    Examples:
        >>> group1 = np.array([1.2, 2.3, 3.1, 4.5, 5.2])
        >>> group2 = np.array([2.1, 3.4, 4.2, 5.5, 6.3])
        >>> result = scirs2.mannwhitneyu_py(group1, group2)
        >>> print(f"U={result['statistic']:.2f}, p={result['pvalue']:.4f}")
    """
    ...

def kruskal_py(*args: NDArray[np.float64]) -> Dict[str, float]:
    """
    Kruskal-Wallis H-test for independent samples.

    Tests whether multiple independent samples have different distributions
    using ranks. This is a nonparametric alternative to one-way ANOVA.

    Parameters:
        *args: Two or more arrays, each representing a group

    Returns:
        Dictionary with:
        - statistic: Kruskal-Wallis H statistic
        - pvalue: p-value for the test

    Raises:
        RuntimeError: If fewer than 2 groups are provided

    Examples:
        >>> group1 = np.array([1.2, 2.3, 3.1, 4.5])
        >>> group2 = np.array([5.1, 6.2, 7.3, 8.4])
        >>> group3 = np.array([9.5, 10.6, 11.7, 12.8])
        >>> result = scirs2.kruskal_py(group1, group2, group3)
        >>> print(f"H={result['statistic']:.2f}, p={result['pvalue']:.4f}")
    """
    ...

# Homogeneity tests
def levene_py(
    *args: NDArray[np.float64],
    center: str = "median",
    proportion_to_cut: float = 0.05,
) -> Dict[str, float]:
    """
    Levene's test for homogeneity of variance.

    Tests the null hypothesis that all input samples are from populations
    with equal variances. More robust than Bartlett's test when data is
    not normally distributed.

    Parameters:
        *args: Two or more arrays, each representing a group
        center: Method to use for center: "mean", "median" (default), or "trimmed"
        proportion_to_cut: Proportion to cut from each end when using "trimmed" (default: 0.05)

    Returns:
        Dictionary with:
        - statistic: Levene test statistic (W)
        - pvalue: p-value for the test

    Raises:
        RuntimeError: If fewer than 2 groups are provided or test fails

    Examples:
        >>> g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        >>> g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])
        >>> g3 = np.array([8.95, 9.12, 8.95, 8.85, 9.03])
        >>> result = scirs2.levene_py(g1, g2, g3)
        >>> print(f"W={result['statistic']:.2f}, p={result['pvalue']:.4f}")

        >>> # Use mean as center
        >>> result = scirs2.levene_py(g1, g2, center="mean")
    """
    ...

def bartlett_test_py(*args: NDArray[np.float64]) -> Dict[str, float]:
    """
    Bartlett's test for homogeneity of variance.

    Tests the null hypothesis that all input samples are from populations
    with equal variances. More powerful than Levene's test but sensitive
    to departures from normality.

    Parameters:
        *args: Two or more arrays, each representing a group

    Returns:
        Dictionary with:
        - statistic: Bartlett test statistic
        - pvalue: p-value for the test

    Raises:
        RuntimeError: If fewer than 2 groups are provided or test fails

    Examples:
        >>> g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        >>> g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])
        >>> result = scirs2.bartlett_test_py(g1, g2)
        >>> print(f"T={result['statistic']:.2f}, p={result['pvalue']:.4f}")
    """
    ...

def brown_forsythe_py(*args: NDArray[np.float64]) -> Dict[str, float]:
    """
    Brown-Forsythe test for homogeneity of variance.

    A modification of Levene's test that uses the median instead of the mean,
    making it more robust against non-normality. Equivalent to Levene's test
    with center="median".

    Parameters:
        *args: Two or more arrays, each representing a group

    Returns:
        Dictionary with:
        - statistic: Brown-Forsythe test statistic
        - pvalue: p-value for the test

    Raises:
        RuntimeError: If fewer than 2 groups are provided or test fails

    Examples:
        >>> g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        >>> g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])
        >>> g3 = np.array([8.95, 9.12, 8.95, 8.85, 9.03])
        >>> result = scirs2.brown_forsythe_py(g1, g2, g3)
        >>> print(f"W={result['statistic']:.2f}, p={result['pvalue']:.4f}")
    """
    ...

# Additional normality tests
def anderson_darling_py(x: NDArray[np.float64]) -> Dict[str, float]:
    """
    Anderson-Darling test for normality.

    Tests whether a sample comes from a normal distribution using the
    Anderson-Darling statistic. This test is more sensitive to deviations
    in the tails of the distribution compared to the Shapiro-Wilk test.

    The Anderson-Darling test statistic quantifies how well the data follows
    a normal distribution, with smaller values indicating better fit.

    Parameters:
        x: Array of sample data (minimum 8 observations)

    Returns:
        Dictionary with:
        - statistic: Anderson-Darling test statistic (A²)
        - pvalue: Two-sided p-value for the test

    Raises:
        RuntimeError: If sample size is less than 8 observations

    Examples:
        >>> data = np.random.normal(0, 1, 100)
        >>> result = scirs2.anderson_darling_py(data)
        >>> print(f"A²={result['statistic']:.3f}, p={result['pvalue']:.4f}")

        >>> # Test non-normal data
        >>> uniform_data = np.random.uniform(-3, 3, 100)
        >>> result = scirs2.anderson_darling_py(uniform_data)
        >>> if result['pvalue'] < 0.05:
        ...     print("Data is not normally distributed")

    Notes:
        The Anderson-Darling test gives more weight to the tails than the
        Kolmogorov-Smirnov test, making it better for detecting departures
        from normality in the tails of the distribution.
    """
    ...

def dagostino_k2_py(x: NDArray[np.float64]) -> Dict[str, float]:
    """
    D'Agostino's K-squared test for normality.

    Tests whether a sample comes from a normal distribution using the
    D'Agostino-Pearson K² test. This test combines skewness and kurtosis
    to provide a comprehensive test for normality.

    The test statistic K² combines standardized measures of skewness and
    kurtosis. Under the null hypothesis of normality, K² approximately
    follows a chi-square distribution with 2 degrees of freedom.

    Parameters:
        x: Array of sample data (minimum 20 observations)

    Returns:
        Dictionary with:
        - statistic: K² test statistic
        - pvalue: Two-sided p-value for the test

    Raises:
        RuntimeError: If sample size is less than 20 observations

    Examples:
        >>> data = np.random.normal(0, 1, 200)
        >>> result = scirs2.dagostino_k2_py(data)
        >>> print(f"K²={result['statistic']:.3f}, p={result['pvalue']:.4f}")

        >>> # Test skewed data
        >>> skewed_data = np.random.exponential(1.0, 200)
        >>> result = scirs2.dagostino_k2_py(skewed_data)
        >>> if result['pvalue'] < 0.05:
        ...     print("Data is not normally distributed")

    Notes:
        This test is particularly sensitive to departures from normality
        due to skewness and kurtosis. It requires a larger minimum sample
        size (n ≥ 20) compared to some other normality tests.

        The test statistic combines:
        - Z(skewness): Standardized measure of skewness
        - Z(kurtosis): Standardized measure of excess kurtosis
        - K² = Z(skewness)² + Z(kurtosis)²
    """
    ...

def ks_2samp_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alternative: str = "two-sided"
) -> Dict[str, float]:
    """
    Two-sample Kolmogorov-Smirnov test.

    Tests whether two independent samples come from the same distribution
    using the Kolmogorov-Smirnov statistic. This is a nonparametric test
    that makes no assumptions about the underlying distributions.

    The test statistic is the maximum absolute difference between the
    empirical cumulative distribution functions (ECDFs) of the two samples.

    Parameters:
        x: First sample array
        y: Second sample array
        alternative: Type of hypothesis test:
            - "two-sided": Test if distributions are different (default)
            - "less": Test if x is stochastically less than y
            - "greater": Test if x is stochastically greater than y

    Returns:
        Dictionary with:
        - statistic: Kolmogorov-Smirnov test statistic (D)
        - pvalue: P-value for the specified alternative hypothesis

    Raises:
        RuntimeError: If either sample is empty

    Examples:
        >>> # Test if two samples come from the same distribution
        >>> x = np.random.normal(0, 1, 100)
        >>> y = np.random.normal(0, 1, 100)
        >>> result = scirs2.ks_2samp_py(x, y)
        >>> print(f"D={result['statistic']:.3f}, p={result['pvalue']:.4f}")

        >>> # Test if distributions are different
        >>> x = np.random.normal(0, 1, 100)
        >>> y = np.random.normal(2, 1, 100)  # Different mean
        >>> result = scirs2.ks_2samp_py(x, y)
        >>> if result['pvalue'] < 0.05:
        ...     print("Distributions are significantly different")

        >>> # One-sided test
        >>> result = scirs2.ks_2samp_py(x, y, alternative="less")
        >>> print(f"One-sided p-value: {result['pvalue']:.4f}")

    Notes:
        - This is a nonparametric test that does not assume normality
        - The test statistic D ranges from 0 to 1
        - Works with samples of different sizes
        - The test is sensitive to differences in both location and shape
    """
    ...

# Repeated measures test
def friedman_py(data: NDArray[np.float64]) -> Dict[str, float]:
    """
    Friedman test for repeated measures.

    Tests whether k treatments (or conditions) have different effects across
    n subjects. This is a nonparametric alternative to repeated measures ANOVA
    that does not assume normality.

    The test ranks the observations within each subject (row) and compares
    the rank sums across treatments (columns). Under the null hypothesis,
    all treatments have the same effect.

    Parameters:
        data: 2D array with shape (n_subjects, k_treatments)
              Each row represents one subject's measurements across all treatments

    Returns:
        Dictionary with:
        - statistic: Friedman test statistic (χ²_r)
        - pvalue: P-value based on chi-square distribution

    Raises:
        RuntimeError: If there are fewer than 2 subjects or 2 treatments

    Examples:
        >>> # Test if 3 treatments have different effects on 5 subjects
        >>> data = np.array([
        ...     [5.1, 4.9, 5.3],  # Subject 1
        ...     [6.2, 5.8, 6.1],  # Subject 2
        ...     [5.7, 5.5, 5.9],  # Subject 3
        ...     [4.8, 4.6, 5.0],  # Subject 4
        ...     [5.3, 5.1, 5.5]   # Subject 5
        ... ])
        >>> result = scirs2.friedman_py(data)
        >>> print(f"χ²={result['statistic']:.3f}, p={result['pvalue']:.4f}")

        >>> # Interpret results
        >>> if result['pvalue'] < 0.05:
        ...     print("Treatments have significantly different effects")
        ... else:
        ...     print("No significant difference between treatments")

    Notes:
        - This is a nonparametric test (does not assume normality)
        - Requires at least 2 subjects and 2 treatments
        - Each row (subject) is ranked independently
        - The test statistic follows approximately a chi-square distribution
          with k-1 degrees of freedom, where k is the number of treatments
        - Use for within-subjects (repeated measures) designs
        - If you reject the null hypothesis, consider post-hoc pairwise
          comparisons to identify which treatments differ
    """
    ...

# Additional statistics
def skew_py(data: NDArray[np.float64]) -> float:
    """Compute skewness of data."""
    ...

def kurtosis_py(data: NDArray[np.float64]) -> float:
    """Compute excess kurtosis of data (Fisher's definition)."""
    ...

def mode_py(data: NDArray[np.float64]) -> float:
    """Compute mode (most frequent value) of data."""
    ...

def gmean_py(data: NDArray[np.float64]) -> float:
    """Compute geometric mean. All values must be positive."""
    ...

def hmean_py(data: NDArray[np.float64]) -> float:
    """Compute harmonic mean. All values must be positive."""
    ...

def zscore_py(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute z-scores (standard scores) for data."""
    ...

# Dispersion and variability measures
def mean_abs_deviation_py(
    data: NDArray[np.float64], center: float | None = None
) -> float:
    """
    Compute mean absolute deviation from a center point.

    Parameters:
        data: Input data array
        center: Center point for deviation calculation. If None, uses the mean.

    Returns:
        Mean absolute deviation

    Examples:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> scirs2.mean_abs_deviation_py(data)  # MAD from mean
        1.2
        >>> scirs2.mean_abs_deviation_py(data, center=3.0)  # MAD from center
        1.2
    """
    ...

def median_abs_deviation_py(
    data: NDArray[np.float64],
    center: float | None = None,
    scale: float | None = None,
) -> float:
    """
    Compute median absolute deviation (robust measure of variability).

    The median absolute deviation is a robust measure of statistical dispersion
    that is more resilient to outliers than standard deviation.

    Parameters:
        data: Input data array
        center: Center point for deviation calculation. If None, uses the median.
        scale: Scale factor to multiply the MAD. Common values:
               - 1.0: Raw MAD (default if None)
               - 1.4826: Approximate standard deviation for normal distribution

    Returns:
        Median absolute deviation (scaled if scale is provided)

    Examples:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> scirs2.median_abs_deviation_py(data)
        1.0
        >>> scirs2.median_abs_deviation_py(data, scale=1.4826)  # Normalized
        1.4826
    """
    ...

def data_range_py(data: NDArray[np.float64]) -> float:
    """
    Compute the range (max - min) of the data.

    Parameters:
        data: Input data array

    Returns:
        Range of the data (maximum value - minimum value)

    Examples:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> scirs2.data_range_py(data)
        4.0
    """
    ...

def coef_variation_py(data: NDArray[np.float64], ddof: int = 1) -> float:
    """
    Compute coefficient of variation (CV = std / mean).

    The coefficient of variation is a unitless measure of relative variability,
    useful for comparing variability across datasets with different scales or units.

    Parameters:
        data: Input data array
        ddof: Delta degrees of freedom for standard deviation calculation (default: 1)

    Returns:
        Coefficient of variation

    Examples:
        >>> data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        >>> scirs2.coef_variation_py(data)  # CV with sample std (ddof=1)
        0.527...
        >>> scirs2.coef_variation_py(data, ddof=0)  # CV with population std
        0.471...
    """
    ...

def gini_coefficient_py(data: NDArray[np.float64]) -> float:
    """
    Compute Gini coefficient (measure of inequality).

    The Gini coefficient measures inequality in a distribution. It ranges from
    0 (perfect equality) to 1 (maximum inequality). Commonly used for income
    and wealth distributions.

    Parameters:
        data: Input data array (must be non-negative)

    Returns:
        Gini coefficient in [0, 1]
        - 0.0: Perfect equality (all values are the same)
        - 1.0: Maximum inequality (one value has everything)

    Examples:
        >>> # Perfect equality
        >>> equal = np.array([1.0, 1.0, 1.0, 1.0])
        >>> scirs2.gini_coefficient_py(equal)
        0.0
        >>> # Some inequality
        >>> unequal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> scirs2.gini_coefficient_py(unequal)
        0.266...
    """
    ...

# Quantile and robust statistics
def boxplot_stats_py(data: NDArray[np.float64], whis: float = 1.5) -> Dict[str, Any]:
    """
    Compute boxplot statistics (five-number summary and outliers).

    Returns quartiles, whiskers, and outliers for creating boxplots and
    identifying extreme values in the data.

    Parameters:
        data: Input data array
        whis: Whisker range factor (default: 1.5). Whiskers extend to the most
              extreme data point within whis * IQR from the quartiles, where
              IQR is the interquartile range (Q3 - Q1).

    Returns:
        Dictionary containing:
        - q1: First quartile (25th percentile)
        - median: Second quartile (50th percentile)
        - q3: Third quartile (75th percentile)
        - whislo: Lower whisker (minimum value within whisker range)
        - whishi: Upper whisker (maximum value within whisker range)
        - outliers: List of values outside the whisker range

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
        >>> stats = scirs2.boxplot_stats_py(data)
        >>> stats["median"]
        6.0
        >>> 100.0 in stats["outliers"]  # Extreme value detected
        True
    """
    ...

def quartiles_py(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute quartiles (Q1, Q2, Q3) of the data.

    Parameters:
        data: Input data array

    Returns:
        Array of three values [Q1, Q2 (median), Q3]

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> q = scirs2.quartiles_py(data)
        >>> len(q)
        3
        >>> q[0] < q[1] < q[2]  # Q1 < Q2 < Q3
        True
    """
    ...

def winsorized_mean_py(data: NDArray[np.float64], limits: float = 0.1) -> float:
    """
    Compute winsorized mean (robust mean).

    The winsorized mean replaces extreme values with less extreme values
    before computing the mean, making it more robust to outliers.

    Parameters:
        data: Input data array
        limits: Proportion of values to winsorize at each end (default: 0.1).
                For example, limits=0.1 replaces the bottom 10% and top 10%
                of values with the values at the 10th and 90th percentiles.

    Returns:
        Winsorized mean

    Examples:
        >>> # Data with outlier
        >>> data = np.array([1, 2, 3, 4, 5, 100])
        >>> regular_mean = np.mean(data)  # 19.17
        >>> robust_mean = scirs2.winsorized_mean_py(data, limits=0.2)
        >>> robust_mean < regular_mean  # Less affected by outlier
        True
    """
    ...

def winsorized_variance_py(
    data: NDArray[np.float64], limits: float = 0.1, ddof: int = 1
) -> float:
    """
    Compute winsorized variance (robust variance).

    The winsorized variance replaces extreme values with less extreme values
    before computing the variance, making it more robust to outliers.

    Parameters:
        data: Input data array
        limits: Proportion of values to winsorize at each end (default: 0.1)
        ddof: Delta degrees of freedom (default: 1 for sample variance)

    Returns:
        Winsorized variance

    Examples:
        >>> # Data with outlier
        >>> data = np.array([1, 2, 3, 4, 5, 100])
        >>> regular_var = np.var(data, ddof=1)  # Large due to outlier
        >>> robust_var = scirs2.winsorized_variance_py(data, limits=0.2, ddof=1)
        >>> robust_var < regular_var / 2  # Much smaller
        True
    """
    ...

# Information theory and advanced statistics
def entropy_py(data: NDArray[np.int64], base: float | None = None) -> float:
    """
    Compute Shannon entropy of discrete data.

    Entropy measures the uncertainty or information content in a discrete
    probability distribution. Higher entropy indicates more uncertainty.

    Parameters:
        data: Input data array (discrete values, e.g., counts or categories)
        base: Logarithm base for entropy calculation (default: e for nats)
              - base=2.0 for bits (information theory)
              - base=e (default) for nats
              - base=10.0 for decimal digits

    Returns:
        Entropy value (non-negative)

    Examples:
        >>> # Uniform distribution: maximum entropy
        >>> data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
        >>> scirs2.entropy_py(data)  # Natural log
        1.0986...
        >>> scirs2.entropy_py(data, base=2.0)  # Bits
        1.584...

        >>> # Deterministic: zero entropy
        >>> certain = np.array([1, 1, 1, 1], dtype=np.int64)
        >>> scirs2.entropy_py(certain)
        0.0
    """
    ...

def kl_divergence_py(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """
    Compute Kullback-Leibler (KL) divergence between two probability distributions.

    KL divergence measures how one probability distribution (q) diverges from
    a reference distribution (p). It's asymmetric: KL(p||q) != KL(q||p).

    Parameters:
        p: First probability distribution (must sum to 1.0)
        q: Second probability distribution (must sum to 1.0)

    Returns:
        KL divergence value (non-negative, zero when p == q)

    Examples:
        >>> # Identical distributions: zero divergence
        >>> p = np.array([0.3, 0.3, 0.4])
        >>> q = np.array([0.3, 0.3, 0.4])
        >>> scirs2.kl_divergence_py(p, q)
        0.0

        >>> # Different distributions: positive divergence
        >>> p = np.array([0.5, 0.3, 0.2])
        >>> q = np.array([0.4, 0.4, 0.2])
        >>> kl = scirs2.kl_divergence_py(p, q)
        >>> kl > 0.0
        True

        >>> # Asymmetric property
        >>> kl_pq = scirs2.kl_divergence_py(p, q)
        >>> kl_qp = scirs2.kl_divergence_py(q, p)
        >>> kl_pq != kl_qp
        True
    """
    ...

def cross_entropy_py(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """
    Compute cross-entropy between two probability distributions.

    Cross-entropy is commonly used as a loss function in machine learning.
    It measures the average number of bits needed to encode samples from p
    using a code optimized for q.

    Parameters:
        p: True probability distribution (must sum to 1.0)
        q: Predicted probability distribution (must sum to 1.0)

    Returns:
        Cross-entropy value (non-negative)

    Note:
        Relationship: H(p,q) = H(p) + KL(p||q)
        where H(p) is entropy of p and KL(p||q) is KL divergence

    Examples:
        >>> # Machine learning classification
        >>> true_label = np.array([0.0, 1.0, 0.0])  # One-hot encoded
        >>> pred_good = np.array([0.1, 0.8, 0.1])  # Good prediction
        >>> pred_poor = np.array([0.6, 0.2, 0.2])  # Poor prediction
        >>> loss_good = scirs2.cross_entropy_py(true_label, pred_good)
        >>> loss_poor = scirs2.cross_entropy_py(true_label, pred_poor)
        >>> loss_good < loss_poor  # Lower loss for better prediction
        True
    """
    ...

def weighted_mean_py(
    data: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    """
    Compute weighted arithmetic mean.

    The weighted mean gives different weights to different values, useful when
    some observations are more important or reliable than others.

    Parameters:
        data: Input data array
        weights: Weight array (same length as data, must be non-negative)

    Returns:
        Weighted mean

    Examples:
        >>> # Equal weights: same as regular mean
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        >>> scirs2.weighted_mean_py(data, weights)
        3.0

        >>> # Emphasize certain values
        >>> data = np.array([10.0, 20.0, 30.0])
        >>> weights = np.array([1.0, 2.0, 3.0])  # Weight last value more
        >>> wm = scirs2.weighted_mean_py(data, weights)
        >>> wm > 20.0  # Closer to 30 than simple mean
        True

        >>> # Portfolio weighted returns
        >>> returns = np.array([5.0, 10.0, -2.0, 8.0])  # Asset returns (%)
        >>> portfolio_weights = np.array([0.4, 0.3, 0.1, 0.2])  # Allocations
        >>> portfolio_return = scirs2.weighted_mean_py(returns, portfolio_weights)
    """
    ...

def moment_py(data: NDArray[np.float64], order: int, center: bool = True) -> float:
    """
    Compute statistical moment of specified order.

    Moments are quantitative measures of the shape of a probability distribution.
    Special cases:
    - 1st moment (uncentered): mean
    - 2nd moment (centered): variance
    - 3rd moment (centered, normalized): skewness
    - 4th moment (centered, normalized): kurtosis

    Parameters:
        data: Input data array
        order: Moment order (positive integer)
        center: If True, compute central moment (about the mean).
                If False, compute raw moment (about zero).

    Returns:
        Moment value

    Examples:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        >>> # First uncentered moment is the mean
        >>> scirs2.moment_py(data, order=1, center=False)
        3.0

        >>> # First central moment is always zero
        >>> scirs2.moment_py(data, order=1, center=True)
        0.0

        >>> # Second central moment is variance
        >>> m2 = scirs2.moment_py(data, order=2, center=True)
        >>> var = np.var(data, ddof=0)
        >>> abs(m2 - var) < 0.001
        True

        >>> # Third central moment (related to skewness)
        >>> m3 = scirs2.moment_py(data, order=3, center=True)

        >>> # Fourth central moment (related to kurtosis)
        >>> m4 = scirs2.moment_py(data, order=4, center=True)
    """
    ...

def quintiles_py(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute quintiles (20th, 40th, 60th, 80th percentiles) of a dataset.

    Quintiles divide the dataset into five equal parts, providing finer-grained
    information than quartiles. Useful for quality control, performance analysis,
    and data stratification.

    Parameters:
        data: Input data array

    Returns:
        Array of 4 quintile values: [Q1, Q2, Q3, Q4]
        - Q1: 20th percentile
        - Q2: 40th percentile
        - Q3: 60th percentile
        - Q4: 80th percentile

    Examples:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        >>> quintiles = scirs2.quintiles_py(data)
        >>> len(quintiles)
        4
        >>> # quintiles are approximately [2.8, 4.6, 6.4, 8.2]

        >>> # Use for quality control zones
        >>> measurements = np.random.normal(100, 5, 500)
        >>> q = scirs2.quintiles_py(measurements)
        >>> # Bottom 20%: below q[0]
        >>> # 20-40%: between q[0] and q[1]
        >>> # 40-60%: between q[1] and q[2]
        >>> # 60-80%: between q[2] and q[3]
        >>> # Top 20%: above q[3]
    """
    ...

def skewness_ci_py(
    data: NDArray[np.float64],
    bias: bool = False,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Compute skewness with bootstrap confidence interval.

    Uses bootstrap resampling to estimate the confidence interval for skewness,
    providing uncertainty quantification for distribution asymmetry measurements.

    Parameters:
        data: Input data array (minimum 3 data points)
        bias: If False, compute bias-corrected skewness estimate
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        seed: Random seed for reproducibility (optional)

    Returns:
        Dictionary with keys:
        - 'estimate': Point estimate of skewness
        - 'lower': Lower bound of confidence interval
        - 'upper': Upper bound of confidence interval
        - 'confidence': Confidence level used

    Examples:
        >>> # Right-skewed data (income distribution)
        >>> incomes = np.random.lognormal(mean=10, sigma=0.5, size=200)
        >>> result = scirs2.skewness_ci_py(incomes, confidence=0.95, seed=42)
        >>> result['estimate'] > 0  # Positive skewness
        True
        >>> result['lower'] < result['estimate'] < result['upper']
        True

        >>> # Symmetric data
        >>> normal_data = np.random.normal(0, 1, 100)
        >>> result = scirs2.skewness_ci_py(normal_data, seed=42)
        >>> abs(result['estimate']) < 0.5  # Near zero for symmetric data
        True

        >>> # Reproducible results with seed
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        >>> r1 = scirs2.skewness_ci_py(data, seed=123)
        >>> r2 = scirs2.skewness_ci_py(data, seed=123)
        >>> r1['estimate'] == r2['estimate']
        True
    """
    ...

def kurtosis_ci_py(
    data: NDArray[np.float64],
    fisher: bool = True,
    bias: bool = False,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Compute kurtosis with bootstrap confidence interval.

    Uses bootstrap resampling to estimate the confidence interval for kurtosis,
    providing uncertainty quantification for tail heaviness measurements.

    Parameters:
        data: Input data array (minimum 4 data points)
        fisher: If True, compute excess kurtosis (Fisher's definition, subtract 3)
                If False, compute raw kurtosis (Pearson's definition)
        bias: If False, compute bias-corrected kurtosis estimate
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        seed: Random seed for reproducibility (optional)

    Returns:
        Dictionary with keys:
        - 'estimate': Point estimate of kurtosis
        - 'lower': Lower bound of confidence interval
        - 'upper': Upper bound of confidence interval
        - 'confidence': Confidence level used

    Note:
        - Fisher's kurtosis (excess kurtosis):
          * Normal distribution has kurtosis = 0
          * Positive values indicate heavy tails (leptokurtic)
          * Negative values indicate light tails (platykurtic)
        - Pearson's kurtosis: Normal distribution has kurtosis = 3

    Examples:
        >>> # Normal-like data
        >>> normal_data = np.random.normal(0, 1, 100)
        >>> result = scirs2.kurtosis_ci_py(normal_data, fisher=True, seed=42)
        >>> abs(result['estimate']) < 1.0  # Near 0 for normal data
        True

        >>> # Heavy-tailed data (with outliers)
        >>> heavy_tail = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, -50])
        >>> result = scirs2.kurtosis_ci_py(heavy_tail, fisher=True, seed=42)
        >>> result['estimate'] > 0  # Positive excess kurtosis
        True

        >>> # Fisher vs Pearson kurtosis
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        >>> fisher_result = scirs2.kurtosis_ci_py(data, fisher=True, seed=42)
        >>> pearson_result = scirs2.kurtosis_ci_py(data, fisher=False, seed=42)
        >>> abs((pearson_result['estimate'] - fisher_result['estimate']) - 3.0) < 0.1
        True

        >>> # Financial returns (often fat-tailed)
        >>> returns = np.concatenate([
        ...     np.random.normal(0, 0.01, 180),
        ...     np.random.normal(0, 0.05, 20)
        ... ])
        >>> result = scirs2.kurtosis_ci_py(returns, fisher=True, seed=42)
        >>> result['estimate'] > 0  # Fat tails indicated by positive excess kurtosis
        True
    """
    ...

def deciles_py(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute deciles (10th, 20th, 30th, ..., 90th percentiles) of a dataset.

    Deciles divide data into 10 equal parts, providing finer granularity than
    quartiles (4 parts) or quintiles (5 parts). They are useful for detailed
    distribution analysis, performance grading, and quality control.

    Parameters:
        data: Input data array

    Returns:
        Array of 9 values representing the 10th through 90th percentiles

    Examples:
        >>> # Basic deciles
        >>> data = np.array([float(i) for i in range(1, 101)])  # 1 to 100
        >>> deciles = scirs2.deciles_py(data)
        >>> len(deciles)
        9
        >>> deciles[0]  # 10th percentile
        10.9
        >>> deciles[4]  # 50th percentile (median)
        50.5
        >>> deciles[8]  # 90th percentile
        90.1

        >>> # Deciles provide finer granularity than quintiles
        >>> data = np.random.normal(100, 15, 500)
        >>> deciles = scirs2.deciles_py(data)
        >>> quintiles = scirs2.quintiles_py(data)
        >>> # Deciles[1] (20th) matches quintiles[0] (20th)
        >>> abs(deciles[1] - quintiles[0]) < 0.5
        True

        >>> # Performance grading with deciles
        >>> scores = np.random.beta(5, 2, 1000) * 100
        >>> deciles = scirs2.deciles_py(scores)
        >>> # Bottom 10%: needs improvement
        >>> # 10-30%: below average
        >>> # 30-70%: average
        >>> # 70-90%: above average
        >>> # Top 10%: excellent
        >>> deciles[0] < deciles[4] < deciles[8]  # Ascending order
        True
    """
    ...

def sem_py(data: NDArray[np.float64], ddof: int = 1) -> float:
    """
    Compute the standard error of the mean (SEM).

    The standard error of the mean measures the variability of the sample mean.
    It is computed as: SEM = std / sqrt(n), where std is the standard deviation
    and n is the sample size. SEM is crucial for constructing confidence intervals
    and hypothesis testing.

    Parameters:
        data: Input data array
        ddof: Degrees of freedom for standard deviation calculation
              - 0 for population standard deviation
              - 1 for sample standard deviation (default)

    Returns:
        Standard error of the mean

    Notes:
        - SEM decreases as sample size increases
        - Used to construct confidence intervals: mean ± z * SEM
        - For 95% CI: mean ± 1.96 * SEM
        - For 99% CI: mean ± 2.576 * SEM

    Examples:
        >>> # Basic SEM calculation
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> sem = scirs2.sem_py(data, ddof=1)
        >>> # SEM ≈ std(data) / sqrt(5) ≈ 1.58 / 2.236 ≈ 0.707
        >>> abs(sem - 0.707) < 0.01
        True

        >>> # SEM decreases with larger sample size
        >>> small_sample = np.random.normal(100, 15, 10)
        >>> large_sample = np.random.normal(100, 15, 1000)
        >>> scirs2.sem_py(large_sample, ddof=1) < scirs2.sem_py(small_sample, ddof=1)
        True

        >>> # Constructing 95% confidence interval
        >>> test_scores = np.array([85.0, 88.0, 92.0, 79.0, 95.0, 87.0, 90.0, 84.0, 91.0, 86.0])
        >>> mean = np.mean(test_scores)
        >>> sem = scirs2.sem_py(test_scores, ddof=1)
        >>> ci_lower = mean - 1.96 * sem
        >>> ci_upper = mean + 1.96 * sem
        >>> ci_lower < mean < ci_upper  # Mean within its own CI
        True

        >>> # ddof parameter effect
        >>> data = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        >>> sem_pop = scirs2.sem_py(data, ddof=0)  # Population
        >>> sem_sample = scirs2.sem_py(data, ddof=1)  # Sample
        >>> sem_sample > sem_pop  # Sample SEM is larger
        True
    """
    ...

def percentile_range_py(
    data: NDArray[np.float64],
    lower_pct: float,
    upper_pct: float,
    interpolation: str = "linear",
) -> float:
    """
    Compute the range between two percentiles.

    This function calculates the difference between an upper and lower percentile,
    providing a measure of spread within a specific portion of the distribution.
    It generalizes the interquartile range (IQR), which is the 75th - 25th percentile.

    Parameters:
        data: Input data array
        lower_pct: Lower percentile (0-100)
        upper_pct: Upper percentile (0-100)
        interpolation: Interpolation method for percentile calculation
                      - "linear": Linear interpolation (default)

    Returns:
        Range between the two percentiles (upper - lower)

    Notes:
        - IQR is equivalent to percentile_range(data, 25, 75)
        - Useful for measuring spread while excluding outliers
        - Common ranges:
          * 10th-90th: Middle 80% spread
          * 5th-95th: Middle 90% spread
          * 25th-75th: Interquartile range (IQR)

    Warning:
        Known issue: May overflow with normal distribution data in some cases.
        Use with uniform or well-behaved distributions for best results.

    Examples:
        >>> # Interquartile range (IQR)
        >>> data = np.array([float(i) for i in range(1, 101)])  # 1 to 100
        >>> iqr_range = scirs2.percentile_range_py(data, 25.0, 75.0)
        >>> abs(iqr_range - 50.0) < 1.0  # Should be approximately 50
        True

        >>> # Custom percentile range
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        >>> # Range between 10th and 90th percentile
        >>> range_10_90 = scirs2.percentile_range_py(data, 10.0, 90.0)
        >>> range_10_90 > 0
        True

        >>> # Full data range (0th to 100th percentile)
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> full_range = scirs2.percentile_range_py(data, 0.0, 100.0)
        >>> abs(full_range - 4.0) < 0.1  # max - min = 5 - 1 = 4
        True

        >>> # Symmetric distribution ranges
        >>> # For symmetric data, 5-50 and 50-95 ranges should be similar
        >>> # (Note: Works best with uniform/well-behaved distributions)

        >>> # Identical values
        >>> constant_data = np.array([42.0, 42.0, 42.0, 42.0, 42.0])
        >>> range_const = scirs2.percentile_range_py(constant_data, 25.0, 75.0)
        >>> abs(range_const) < 1e-10  # Should be 0
        True
    """
    ...

def skewness_simd_py(data: NDArray[np.float64], bias: bool = False) -> float:
    """
    Compute SIMD-optimized skewness (third standardized moment).

    This is a high-performance implementation of skewness calculation using SIMD
    (Single Instruction, Multiple Data) acceleration for improved performance on
    large datasets. Skewness measures the asymmetry of a distribution around its mean.

    The skewness formula is: g₁ = E[(X-μ)³] / σ³

    where μ is the mean and σ is the standard deviation.

    Parameters:
        data: Input data array
        bias: If True, use biased estimator (default: False)
              If False, apply sample bias correction (requires n >= 3)

    Returns:
        Skewness value:
        - 0: Symmetric distribution (e.g., normal distribution)
        - > 0: Positively skewed (right tail longer)
        - < 0: Negatively skewed (left tail longer)

    Performance:
        - Uses SIMD acceleration for arrays larger than threshold
        - Significantly faster than regular skew_py for large datasets
        - Automatic fallback to scalar computation for small arrays

    Examples:
        >>> # Symmetric distribution
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> skew = scirs2.skewness_simd_py(data, bias=True)
        >>> abs(skew) < 1e-10  # Near zero for symmetric data
        True

        >>> # Positively skewed (right tail)
        >>> right_skewed = np.array([1.0, 2.0, 2.0, 3.0, 10.0])
        >>> skew_pos = scirs2.skewness_simd_py(right_skewed)
        >>> skew_pos > 0
        True

        >>> # Negatively skewed (left tail)
        >>> left_skewed = np.array([1.0, 8.0, 8.0, 9.0, 10.0])
        >>> skew_neg = scirs2.skewness_simd_py(left_skewed)
        >>> skew_neg < 0
        True

        >>> # Large array with SIMD optimization
        >>> np.random.seed(42)
        >>> large_data = np.random.normal(0, 1, 100000)
        >>> skew_large = scirs2.skewness_simd_py(large_data, bias=False)
        >>> abs(skew_large) < 0.05  # Near zero for normal distribution
        True
    """
    ...

def kurtosis_simd_py(
    data: NDArray[np.float64], fisher: bool = True, bias: bool = False
) -> float:
    """
    Compute SIMD-optimized kurtosis (fourth standardized moment).

    This is a high-performance implementation of kurtosis calculation using SIMD
    acceleration. Kurtosis measures the "tailedness" of a distribution - how prone
    it is to producing outliers.

    The kurtosis formula is: g₂ = E[(X-μ)⁴] / σ⁴

    where μ is the mean and σ is the standard deviation.

    Parameters:
        data: Input data array
        fisher: If True, use Fisher's definition (excess kurtosis: subtract 3)
                If False, use Pearson's definition (raw kurtosis)
                Default: True
        bias: If True, use biased estimator
              If False, apply sample bias correction (requires n >= 4)
              Default: False

    Returns:
        Kurtosis value:
        - Fisher's (excess kurtosis):
          * 0: Normal distribution (mesokurtic)
          * > 0: Heavy tails, more outliers (leptokurtic)
          * < 0: Light tails, fewer outliers (platykurtic)
        - Pearson's (raw kurtosis):
          * 3: Normal distribution
          * > 3: Heavy tails
          * < 3: Light tails

    Performance:
        - Uses SIMD acceleration for arrays larger than threshold
        - Significantly faster than regular kurtosis_py for large datasets
        - Automatic fallback to scalar computation for small arrays

    Examples:
        >>> # Normal-like distribution (Fisher's excess kurtosis ≈ 0)
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        >>> kurt_fisher = scirs2.kurtosis_simd_py(data, fisher=True, bias=True)
        >>> kurt_fisher < 0  # Uniform has negative excess kurtosis
        True

        >>> # Pearson's definition (Fisher's + 3)
        >>> kurt_pearson = scirs2.kurtosis_simd_py(data, fisher=False, bias=True)
        >>> abs(kurt_pearson - (kurt_fisher + 3.0)) < 1e-10
        True

        >>> # Peaked distribution (high kurtosis)
        >>> peaked = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 15.0, 5.0, 5.0])
        >>> kurt_peaked = scirs2.kurtosis_simd_py(peaked, fisher=True)
        >>> kurt_peaked > 0  # Positive excess kurtosis
        True

        >>> # Large array with SIMD optimization
        >>> np.random.seed(42)
        >>> large_data = np.random.normal(0, 1, 100000)
        >>> kurt_large = scirs2.kurtosis_simd_py(large_data, fisher=True, bias=False)
        >>> abs(kurt_large) < 0.2  # Near zero for normal distribution
        True
    """
    ...

def pearson_r_simd_py(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """
    Compute SIMD-optimized Pearson correlation coefficient.

    This is a high-performance implementation using SIMD acceleration for computing
    the Pearson correlation coefficient, which measures the linear relationship
    between two variables.

    The Pearson correlation formula is:
    r = Σ((x-μₓ)(y-μᵧ)) / √(Σ(x-μₓ)² × Σ(y-μᵧ)²)

    where μₓ and μᵧ are the means of x and y.

    Parameters:
        x: First data array
        y: Second data array (must have same length as x)

    Returns:
        Correlation coefficient ranging from -1 to 1:
        - 1: Perfect positive correlation
        - 0: No linear correlation
        - -1: Perfect negative correlation

    Raises:
        RuntimeError: If arrays have different lengths or zero variance

    Performance:
        - Uses SIMD acceleration for arrays larger than threshold
        - Significantly faster than regular pearsonr_py for large datasets
        - Automatic fallback to scalar computation for small arrays

    Examples:
        >>> # Perfect positive correlation
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> corr = scirs2.pearson_r_simd_py(x, y)
        >>> abs(corr - 1.0) < 1e-10
        True

        >>> # Perfect negative correlation
        >>> y_neg = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        >>> corr_neg = scirs2.pearson_r_simd_py(x, y_neg)
        >>> abs(corr_neg - (-1.0)) < 1e-10
        True

        >>> # No correlation
        >>> np.random.seed(42)
        >>> x_rand = np.random.normal(0, 1, 1000)
        >>> y_rand = np.random.normal(0, 1, 1000)
        >>> corr_none = scirs2.pearson_r_simd_py(x_rand, y_rand)
        >>> abs(corr_none) < 0.1  # Close to zero
        True

        >>> # Moderate positive correlation
        >>> np.random.seed(42)
        >>> x_data = np.random.normal(0, 1, 100)
        >>> y_data = 0.7 * x_data + np.random.normal(0, 0.5, 100)
        >>> corr_mod = scirs2.pearson_r_simd_py(x_data, y_data)
        >>> 0.5 < corr_mod < 0.9
        True

        >>> # Financial data (stock returns)
        >>> np.random.seed(42)
        >>> stock_a = np.random.normal(0.001, 0.02, 252)  # Daily returns
        >>> stock_b = 0.6 * stock_a + np.random.normal(0.0005, 0.015, 252)
        >>> corr_stocks = scirs2.pearson_r_simd_py(stock_a, stock_b)
        >>> 0.4 < corr_stocks < 0.8
        True
    """
    ...

def covariance_simd_py(
    x: NDArray[np.float64], y: NDArray[np.float64], ddof: int = 1
) -> float:
    """
    Compute SIMD-optimized covariance.

    This is a high-performance implementation using SIMD acceleration for computing
    covariance, which measures how two variables change together.

    The covariance formula is:
    Cov(X,Y) = Σ((x-μₓ)(y-μᵧ)) / (n - ddof)

    where μₓ and μᵧ are the means of x and y, and ddof is degrees of freedom.

    Parameters:
        x: First data array
        y: Second data array (must have same length as x)
        ddof: Degrees of freedom for bias correction
              - 0: Population covariance (biased)
              - 1: Sample covariance (unbiased, default)

    Returns:
        Covariance value:
        - > 0: Variables tend to increase together
        - 0: No linear relationship
        - < 0: Variables move in opposite directions

    Raises:
        RuntimeError: If arrays have different lengths or insufficient data

    Performance:
        - Uses SIMD acceleration for arrays larger than threshold
        - Significantly faster than regular covariance_py for large datasets
        - Automatic fallback to scalar computation for small arrays

    Notes:
        - Covariance is related to correlation: Cov(X,Y) = r × σₓ × σᵧ
        - Units: product of the units of x and y
        - Covariance is scale-dependent (unlike correlation)

    Examples:
        >>> # Positive covariance (variables increase together)
        >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> cov = scirs2.covariance_simd_py(x, y, ddof=1)
        >>> cov > 0
        True

        >>> # Negative covariance (variables move opposite)
        >>> y_neg = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        >>> cov_neg = scirs2.covariance_simd_py(x, y_neg, ddof=1)
        >>> cov_neg < 0
        True

        >>> # Zero covariance (no relationship)
        >>> y_const = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        >>> cov_zero = scirs2.covariance_simd_py(x, y_const, ddof=1)
        >>> abs(cov_zero) < 1e-10
        True

        >>> # Formula verification: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)]
        >>> x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        >>> cov_result = scirs2.covariance_simd_py(x_data, y_data, ddof=1)
        >>> # Manual calculation
        >>> mean_x = np.mean(x_data)
        >>> mean_y = np.mean(y_data)
        >>> manual_cov = np.sum((x_data - mean_x) * (y_data - mean_y)) / 4
        >>> abs(cov_result - manual_cov) < 1e-10
        True

        >>> # ddof=0 (population) vs ddof=1 (sample)
        >>> cov_pop = scirs2.covariance_simd_py(x_data, y_data, ddof=0)
        >>> cov_sample = scirs2.covariance_simd_py(x_data, y_data, ddof=1)
        >>> cov_pop < cov_sample  # Population cov is smaller
        True

        >>> # Large arrays with SIMD optimization
        >>> np.random.seed(42)
        >>> large_x = np.random.normal(0, 1, 100000)
        >>> large_y = 0.7 * large_x + np.random.normal(0, 0.5, 100000)
        >>> cov_large = scirs2.covariance_simd_py(large_x, large_y, ddof=1)
        >>> cov_large > 0  # Positive relationship
        True
    """
    ...

def moment_simd_py(
    data: NDArray[np.float64], moment_order: int, center: bool = True
) -> float:
    """
    Compute SIMD-optimized nth statistical moment.

    This is a high-performance implementation for calculating any statistical moment
    using SIMD acceleration. Moments are fundamental in describing distributions:
    - 1st moment (raw): Mean
    - 2nd moment (central): Variance (with ddof=0)
    - 3rd moment (central): Related to skewness
    - 4th moment (central): Related to kurtosis

    Formula:
    - Raw moment: E[X^n] = Σ(x^n) / N
    - Central moment: E[(X-μ)^n] = Σ((x-μ)^n) / N

    Parameters:
        data: Input data array
        moment_order: Order of the moment (0, 1, 2, 3, ...)
        center: If True, compute central moment (around mean)
                If False, compute raw moment (default: True)

    Returns:
        The nth moment value

    Performance:
        - Uses SIMD acceleration for large arrays
        - Efficient computation of arbitrary moment orders
        - Automatic optimization selection

    Examples:
        >>> # First raw moment = mean
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> moment1_raw = scirs2.moment_simd_py(data, 1, center=False)
        >>> abs(moment1_raw - 3.0) < 1e-10
        True

        >>> # First central moment = 0
        >>> moment1_central = scirs2.moment_simd_py(data, 1, center=True)
        >>> abs(moment1_central) < 1e-10
        True

        >>> # Second central moment = variance (ddof=0)
        >>> moment2 = scirs2.moment_simd_py(data, 2, center=True)
        >>> variance_pop = np.var(data, ddof=0)
        >>> abs(moment2 - variance_pop) < 1e-10
        True

        >>> # Third central moment (symmetry measure)
        >>> moment3 = scirs2.moment_simd_py(data, 3, center=True)
        >>> abs(moment3) < 1e-10  # Symmetric data
        True

        >>> # Fourth central moment (tail heaviness)
        >>> moment4 = scirs2.moment_simd_py(data, 4, center=True)
        >>> moment4 > 0  # Always positive
        True

        >>> # Gamma distribution moments
        >>> np.random.seed(42)
        >>> gamma_data = np.random.gamma(5.0, 2.0, 10000)
        >>> m1 = scirs2.moment_simd_py(gamma_data, 1, center=False)
        >>> 9.5 < m1 < 10.5  # Mean ≈ k*theta = 10
        True
    """
    ...

def mean_simd_py(data: NDArray[np.float64]) -> float:
    """
    Compute SIMD-optimized arithmetic mean.

    This is a high-performance implementation of the mean calculation using SIMD
    acceleration, providing significant speedup for large datasets while maintaining
    numerical accuracy.

    Formula: μ = Σx / n

    Parameters:
        data: Input data array

    Returns:
        The arithmetic mean of the data

    Performance:
        - Uses SIMD acceleration for large arrays (typically >1000 elements)
        - 2-8x faster than scalar computation depending on CPU
        - Automatic fallback for small arrays
        - Platform-aware: Leverages SSE, AVX, or NEON instructions

    Examples:
        >>> # Basic mean
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> mean = scirs2.mean_simd_py(data)
        >>> mean == 3.0
        True

        >>> # Mean with negative values
        >>> data_neg = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        >>> mean_neg = scirs2.mean_simd_py(data_neg)
        >>> abs(mean_neg) < 1e-10
        True

        >>> # Large array SIMD optimization
        >>> np.random.seed(42)
        >>> large_data = np.random.normal(100, 15, 100000)
        >>> mean_large = scirs2.mean_simd_py(large_data)
        >>> 99 < mean_large < 101
        True

        >>> # Single value
        >>> single = np.array([42.0])
        >>> scirs2.mean_simd_py(single)
        42.0

        >>> # Matches NumPy and regular version
        >>> test_data = np.array([10.5, 20.3, 30.7, 40.2, 50.1])
        >>> simd_mean = scirs2.mean_simd_py(test_data)
        >>> numpy_mean = np.mean(test_data)
        >>> regular_mean = scirs2.mean_py(test_data)
        >>> abs(simd_mean - numpy_mean) < 1e-10
        True
        >>> abs(simd_mean - regular_mean) < 1e-10
        True
    """
    ...

def std_simd_py(data: NDArray[np.float64], ddof: int = 1) -> float:
    """
    Compute SIMD-optimized standard deviation.

    This is a high-performance implementation of standard deviation using SIMD
    acceleration. Standard deviation measures the amount of variation or dispersion
    in a dataset.

    Formula: σ = √(Σ(x-μ)² / (n-ddof))

    Parameters:
        data: Input data array
        ddof: Degrees of freedom for bias correction
              - 0: Population standard deviation (biased)
              - 1: Sample standard deviation (unbiased, default)

    Returns:
        The standard deviation of the data

    Performance:
        - Uses SIMD acceleration for large arrays
        - 3-8x faster than scalar computation
        - Numerically stable Welford-based algorithm
        - Automatic optimization selection

    Examples:
        >>> # Sample standard deviation
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> std_sample = scirs2.std_simd_py(data, ddof=1)
        >>> expected = np.std(data, ddof=1)
        >>> abs(std_sample - expected) < 1e-10
        True

        >>> # Population vs sample std
        >>> pop_std = scirs2.std_simd_py(data, ddof=0)
        >>> sample_std = scirs2.std_simd_py(data, ddof=1)
        >>> pop_std < sample_std  # Population is smaller
        True

        >>> # Constant data has zero std
        >>> constant = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        >>> std_zero = scirs2.std_simd_py(constant, ddof=1)
        >>> abs(std_zero) < 1e-10
        True

        >>> # Large array optimization
        >>> np.random.seed(42)
        >>> large_data = np.random.normal(0, 5, 100000)
        >>> std_large = scirs2.std_simd_py(large_data, ddof=1)
        >>> 4.9 < std_large < 5.1  # Close to true value (5)
        True

        >>> # Financial volatility
        >>> np.random.seed(42)
        >>> returns = np.random.normal(0.0005, 0.0127, 252)
        >>> volatility = scirs2.std_simd_py(returns, ddof=1)
        >>> 0.01 < volatility < 0.02
        True
    """
    ...

def variance_simd_py(data: NDArray[np.float64], ddof: int = 1) -> float:
    """
    Compute SIMD-optimized variance.

    This is a high-performance implementation of variance using SIMD acceleration.
    Variance measures the average squared deviation from the mean, quantifying
    the spread of data.

    Formula: σ² = Σ(x-μ)² / (n-ddof)

    Parameters:
        data: Input data array
        ddof: Degrees of freedom for bias correction
              - 0: Population variance (biased)
              - 1: Sample variance (unbiased, default)

    Returns:
        The variance of the data

    Performance:
        - Uses SIMD acceleration for large arrays
        - 3-8x faster than scalar computation
        - Numerically stable algorithm
        - Automatic optimization selection

    Notes:
        - Variance = std²
        - Units: squared units of the data
        - Related to 2nd central moment: variance = moment₂ with ddof adjustment

    Examples:
        >>> # Sample variance
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> var_sample = scirs2.variance_simd_py(data, ddof=1)
        >>> expected = np.var(data, ddof=1)
        >>> abs(var_sample - expected) < 1e-10
        True

        >>> # Population variance (smaller than sample)
        >>> var_pop = scirs2.variance_simd_py(data, ddof=0)
        >>> var_pop == 2.0
        True
        >>> var_sample == 2.5
        True

        >>> # Variance = std²
        >>> variance = scirs2.variance_simd_py(data, ddof=1)
        >>> std = scirs2.std_simd_py(data, ddof=1)
        >>> abs(variance - std**2) < 1e-10
        True

        >>> # Constant data has zero variance
        >>> constant = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        >>> var_zero = scirs2.variance_simd_py(constant, ddof=1)
        >>> abs(var_zero) < 1e-10
        True

        >>> # Large array optimization
        >>> np.random.seed(42)
        >>> large_data = np.random.normal(0, 3, 100000)
        >>> var_large = scirs2.variance_simd_py(large_data, ddof=1)
        >>> 8.8 < var_large < 9.2  # Close to 9 (3²)
        True

        >>> # Sensor measurement precision
        >>> np.random.seed(42)
        >>> measurements = np.random.normal(25.0, 0.5, 1000)
        >>> var_sensor = scirs2.variance_simd_py(measurements, ddof=1)
        >>> 0.20 < var_sensor < 0.30  # Close to 0.25 (0.5²)
        True
    """
    ...

# Statistical Distributions
# =============================================================================

class norm:
    """
    Normal (Gaussian) distribution.

    Provides PDF, CDF, PPF, and random variate generation for the normal distribution.

    Examples:
        >>> # Standard normal distribution (mean=0, std=1)
        >>> dist = scirs2.norm()
        >>> pdf_value = dist.pdf(0.0)  # ~0.3989
        >>> cdf_value = dist.cdf(0.0)  # 0.5
        >>> median = dist.ppf(0.5)     # 0.0
        >>> samples = dist.rvs(1000)   # 1000 random samples
        >>>
        >>> # Custom normal distribution
        >>> dist2 = scirs2.norm(loc=5.0, scale=2.0)  # mean=5, std=2
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create a normal distribution.

        Parameters:
            loc: Mean (location parameter)
            scale: Standard deviation (scale parameter), must be > 0

        Raises:
            RuntimeError: If scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate the PDF

        Returns:
            PDF value at x
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate the CDF

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q

        Raises:
            RuntimeError: If q not in [0, 1]
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples to generate

        Returns:
            List of random samples from the distribution
        """
        ...

class binom:
    """
    Binomial distribution.

    Models the number of successes in n independent Bernoulli trials,
    each with probability p of success.

    Examples:
        >>> # 10 coin flips, fair coin
        >>> dist = scirs2.binom(10, 0.5)
        >>> pmf_5 = dist.pmf(5.0)  # Probability of exactly 5 heads
        >>> cdf_7 = dist.cdf(7.0)  # Probability of 7 or fewer heads
        >>> median = dist.ppf(0.5)  # Median number of successes
        >>> samples = dist.rvs(1000)  # 1000 random samples
    """

    def __init__(self, n: int, p: float) -> None:
        """
        Create a binomial distribution.

        Parameters:
            n: Number of trials (must be >= 0)
            p: Probability of success (must be in [0, 1])

        Raises:
            RuntimeError: If n < 0 or p not in [0, 1]
        """
        ...

    def pmf(self, k: float) -> float:
        """
        Probability mass function.

        Parameters:
            k: Number of successes

        Returns:
            Probability of exactly k successes
        """
        ...

    def cdf(self, k: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            k: Number of successes

        Returns:
            Probability of k or fewer successes
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            Smallest k such that CDF(k) >= q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples
        """
        ...

class poisson:
    """
    Poisson distribution.

    Models the number of events occurring in a fixed interval when events
    occur independently at a constant average rate.

    Examples:
        >>> # Average 3 events per interval
        >>> dist = scirs2.poisson(3.0)
        >>> pmf_3 = dist.pmf(3.0)  # Probability of exactly 3 events
        >>> cdf_5 = dist.cdf(5.0)  # Probability of 5 or fewer events
        >>> samples = dist.rvs(1000)  # 1000 random samples
    """

    def __init__(self, mu: float) -> None:
        """
        Create a Poisson distribution.

        Parameters:
            mu: Expected number of events (lambda parameter), must be > 0

        Raises:
            RuntimeError: If mu <= 0
        """
        ...

    def pmf(self, k: float) -> float:
        """
        Probability mass function.

        Parameters:
            k: Number of events

        Returns:
            Probability of exactly k events
        """
        ...

    def cdf(self, k: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            k: Number of events

        Returns:
            Probability of k or fewer events
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF / quantile function).

        Returns the smallest integer k such that CDF(k) >= q.

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            Smallest k such that CDF(k) >= q

        Raises:
            RuntimeError: If q is outside [0, 1] or the computation
                otherwise fails.
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples
        """
        ...

class expon:
    """
    Exponential distribution.

    Models the time between events in a Poisson process (time until next event).

    Examples:
        >>> # Default: mean=1
        >>> dist = scirs2.expon()
        >>> pdf_1 = dist.pdf(1.0)
        >>> cdf_2 = dist.cdf(2.0)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom scale (mean = scale)
        >>> dist2 = scirs2.expon(scale=2.0)  # mean=2
    """

    def __init__(self, scale: float = 1.0) -> None:
        """
        Create an exponential distribution.

        Parameters:
            scale: Scale parameter (mean = scale), must be > 0

        Raises:
            RuntimeError: If scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value at x
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples
        """
        ...

class uniform:
    """
    Uniform distribution.

    Constant probability over the interval [loc, loc+scale).

    Examples:
        >>> # Standard uniform [0, 1)
        >>> dist = scirs2.uniform()
        >>> pdf_0_5 = dist.pdf(0.5)  # = 1.0
        >>> cdf_0_5 = dist.cdf(0.5)  # = 0.5
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom range [2, 5)
        >>> dist2 = scirs2.uniform(loc=2.0, scale=3.0)
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create a uniform distribution over [loc, loc+scale).

        Parameters:
            loc: Lower bound
            scale: Width of the interval, must be > 0

        Raises:
            RuntimeError: If scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value (1/scale if in [loc, loc+scale), else 0)
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples uniformly distributed over [loc, loc+scale)
        """
        ...

class beta:
    """
    Beta distribution.

    The beta distribution is a continuous probability distribution defined on
    the interval [0, 1], parameterized by two positive shape parameters alpha
    and beta.

    Examples:
        >>> # Standard beta distribution
        >>> dist = scirs2.beta(2.0, 3.0)
        >>> pdf_0_5 = dist.pdf(0.5)
        >>> cdf_0_5 = dist.cdf(0.5)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Shifted and scaled beta
        >>> dist2 = scirs2.beta(alpha=2.0, beta=5.0, loc=1.0, scale=2.0)
    """

    def __init__(self, alpha: float, beta: float, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create a beta distribution.

        Parameters:
            alpha: Shape parameter alpha > 0
            beta: Shape parameter beta > 0
            loc: Location parameter (default: 0.0)
            scale: Scale parameter, must be > 0 (default: 1.0)

        Raises:
            RuntimeError: If alpha <= 0, beta <= 0, or scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the beta distribution
        """
        ...

class gamma:
    """
    Gamma distribution.

    The gamma distribution is a continuous probability distribution with
    shape parameter k and scale parameter theta.

    Examples:
        >>> # Standard gamma distribution
        >>> dist = scirs2.gamma(2.0)
        >>> pdf_1 = dist.pdf(1.0)
        >>> cdf_1 = dist.cdf(1.0)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom parameters
        >>> dist2 = scirs2.gamma(shape=2.0, scale=2.0, loc=0.5)
    """

    def __init__(self, shape: float, scale: float = 1.0, loc: float = 0.0) -> None:
        """
        Create a gamma distribution.

        Parameters:
            shape: Shape parameter k > 0
            scale: Scale parameter theta > 0 (default: 1.0)
            loc: Location parameter (default: 0.0)

        Raises:
            RuntimeError: If shape <= 0 or scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the gamma distribution
        """
        ...

class chi2:
    """
    Chi-square distribution.

    The chi-square distribution with k degrees of freedom is the distribution
    of a sum of the squares of k independent standard normal random variables.

    Examples:
        >>> # Chi-square with 5 degrees of freedom
        >>> dist = scirs2.chi2(5.0)
        >>> pdf_3 = dist.pdf(3.0)
        >>> cdf_3 = dist.cdf(3.0)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # With location and scale
        >>> dist2 = scirs2.chi2(df=10.0, loc=1.0, scale=2.0)
    """

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create a chi-square distribution.

        Parameters:
            df: Degrees of freedom > 0
            loc: Location parameter (default: 0.0)
            scale: Scale parameter, must be > 0 (default: 1.0)

        Raises:
            RuntimeError: If df <= 0 or scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Note: Current implementation has known numerical issues.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the chi-square distribution
        """
        ...

class t:
    """
    Student's t distribution.

    The Student's t distribution is a continuous probability distribution that
    arises when estimating the mean of a normally distributed population in
    situations where the sample size is small.

    Examples:
        >>> # Standard t distribution with 5 df
        >>> dist = scirs2.t(5.0)
        >>> pdf_0 = dist.pdf(0.0)  # Symmetric around 0
        >>> cdf_0 = dist.cdf(0.0)  # = 0.5
        >>> samples = dist.rvs(1000)
        >>>
        >>> # With location and scale
        >>> dist2 = scirs2.t(df=10.0, loc=1.0, scale=2.0)
    """

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create a Student's t distribution.

        Parameters:
            df: Degrees of freedom > 0
            loc: Location parameter (default: 0.0)
            scale: Scale parameter, must be > 0 (default: 1.0)

        Raises:
            RuntimeError: If df <= 0 or scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the Student's t distribution
        """
        ...

class cauchy:
    """
    Cauchy (Lorentz) distribution.

    The Cauchy distribution is a continuous probability distribution with
    heavy tails. It has no defined mean or variance.

    Examples:
        >>> # Standard Cauchy distribution
        >>> dist = scirs2.cauchy()
        >>> pdf_0 = dist.pdf(0.0)  # = 1/pi
        >>> cdf_0 = dist.cdf(0.0)  # = 0.5
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom location and scale
        >>> dist2 = scirs2.cauchy(loc=1.0, scale=2.0)
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create a Cauchy distribution.

        Parameters:
            loc: Location parameter (median) (default: 0.0)
            scale: Scale parameter, must be > 0 (default: 1.0)

        Raises:
            RuntimeError: If scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the Cauchy distribution
        """
        ...

class f:
    """
    F (Fisher-Snedecor) distribution.

    The F distribution is a continuous probability distribution that arises
    in the analysis of variance (ANOVA) and in F-tests.

    Examples:
        >>> # F distribution with 5 and 10 degrees of freedom
        >>> dist = scirs2.f(5.0, 10.0)
        >>> pdf_1 = dist.pdf(1.0)
        >>> cdf_1 = dist.cdf(1.0)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # With location and scale
        >>> dist2 = scirs2.f(dfn=2.0, dfd=20.0, loc=0.5, scale=2.0)
    """

    def __init__(self, dfn: float, dfd: float, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create an F distribution.

        Parameters:
            dfn: Numerator degrees of freedom > 0
            dfd: Denominator degrees of freedom > 0
            loc: Location parameter (default: 0.0)
            scale: Scale parameter, must be > 0 (default: 1.0)

        Raises:
            RuntimeError: If dfn <= 0, dfd <= 0, or scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the F distribution
        """
        ...

class lognorm:
    """
    Lognormal distribution.

    The lognormal distribution is the distribution of a random variable whose
    logarithm follows a normal distribution. Commonly used in finance, biology,
    and other fields where quantities are multiplicative in nature.

    Examples:
        >>> # Standard lognormal distribution
        >>> dist = scirs2.lognorm()
        >>> pdf_1 = dist.pdf(1.0)
        >>> cdf_1 = dist.cdf(1.0)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom parameters
        >>> dist2 = scirs2.lognorm(mu=0.5, sigma=0.8, loc=0.0)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, loc: float = 0.0) -> None:
        """
        Create a lognormal distribution.

        Parameters:
            mu: Mean of underlying normal distribution (default: 0.0)
            sigma: Standard deviation of underlying normal distribution > 0 (default: 1.0)
            loc: Location parameter (default: 0.0)

        Raises:
            RuntimeError: If sigma <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value (0 if x <= loc)
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the lognormal distribution
        """
        ...

class weibull_min:
    """
    Weibull minimum distribution.

    The Weibull distribution is widely used in reliability engineering and
    failure analysis. It can model a variety of life data behaviors depending
    on the value of the shape parameter.

    Examples:
        >>> # Weibull distribution with shape=2 (Rayleigh-like)
        >>> dist = scirs2.weibull_min(2.0)
        >>> pdf_1 = dist.pdf(1.0)
        >>> cdf_1 = dist.cdf(1.0)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom parameters
        >>> dist2 = scirs2.weibull_min(shape=1.5, scale=2.0, loc=0.0)
    """

    def __init__(self, shape: float, scale: float = 1.0, loc: float = 0.0) -> None:
        """
        Create a Weibull distribution.

        Parameters:
            shape: Shape parameter k > 0
            scale: Scale parameter lambda > 0 (default: 1.0)
            loc: Location parameter (default: 0.0)

        Raises:
            RuntimeError: If shape <= 0 or scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value (0 if x <= loc)
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the Weibull distribution
        """
        ...

class laplace:
    """
    Laplace (double exponential) distribution.

    The Laplace distribution is symmetric with heavier tails than the normal
    distribution. Used in robust statistics and signal processing.

    Examples:
        >>> # Standard Laplace distribution
        >>> dist = scirs2.laplace()
        >>> pdf_0 = dist.pdf(0.0)  # = 0.5
        >>> cdf_0 = dist.cdf(0.0)  # = 0.5
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom location and scale
        >>> dist2 = scirs2.laplace(loc=2.0, scale=3.0)
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create a Laplace distribution.

        Parameters:
            loc: Location parameter (median) (default: 0.0)
            scale: Scale parameter > 0 (default: 1.0)

        Raises:
            RuntimeError: If scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the Laplace distribution
        """
        ...

class logistic:
    """
    Logistic distribution.

    The logistic distribution is used in growth models, neural networks, and
    logistic regression. It resembles the normal distribution but has heavier
    tails.

    Examples:
        >>> # Standard logistic distribution
        >>> dist = scirs2.logistic()
        >>> pdf_0 = dist.pdf(0.0)  # = 0.25
        >>> cdf_0 = dist.cdf(0.0)  # = 0.5
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom location and scale
        >>> dist2 = scirs2.logistic(loc=1.0, scale=2.0)
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        """
        Create a logistic distribution.

        Parameters:
            loc: Location parameter (mean, median, mode) (default: 0.0)
            scale: Scale parameter > 0 (default: 1.0)

        Raises:
            RuntimeError: If scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the logistic distribution
        """
        ...

class pareto:
    """
    Pareto distribution.

    The Pareto distribution is a power-law distribution commonly used to model
    wealth distribution, city populations, and other phenomena following the
    "80-20 rule" (Pareto principle).

    Examples:
        >>> # Pareto distribution with shape=3
        >>> dist = scirs2.pareto(3.0)
        >>> pdf_2 = dist.pdf(2.0)
        >>> cdf_2 = dist.cdf(2.0)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Custom parameters
        >>> dist2 = scirs2.pareto(shape=2.0, scale=2.0, loc=0.0)
    """

    def __init__(self, shape: float, scale: float = 1.0, loc: float = 0.0) -> None:
        """
        Create a Pareto distribution.

        Parameters:
            shape: Shape parameter alpha > 0
            scale: Scale parameter x_m > 0 (default: 1.0)
            loc: Location parameter (default: 0.0)

        Raises:
            RuntimeError: If shape <= 0 or scale <= 0
        """
        ...

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            PDF value (0 if x <= scale+loc)
        """
        ...

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Probability that X <= x
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            x such that CDF(x) = q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples from the Pareto distribution
        """
        ...

class geom:
    """
    Geometric distribution.

    The geometric distribution models the number of failures before the first
    success in a sequence of independent Bernoulli trials. Commonly used in
    reliability analysis and queuing theory.

    Examples:
        >>> # Geometric distribution with p=0.3
        >>> dist = scirs2.geom(0.3)
        >>> pmf_2 = dist.pmf(2.0)  # P(X = 2)
        >>> cdf_2 = dist.cdf(2.0)  # P(X <= 2)
        >>> samples = dist.rvs(1000)
        >>>
        >>> # Geometric with p=0.5 (fair coin)
        >>> dist2 = scirs2.geom(0.5)
    """

    def __init__(self, p: float) -> None:
        """
        Create a geometric distribution.

        Parameters:
            p: Success probability, 0 < p <= 1

        Raises:
            RuntimeError: If p <= 0 or p > 1
        """
        ...

    def pmf(self, k: float) -> float:
        """
        Probability mass function.

        Parameters:
            k: Number of failures before first success (non-negative integer)

        Returns:
            Probability P(X = k)
        """
        ...

    def cdf(self, k: float) -> float:
        """
        Cumulative distribution function.

        Parameters:
            k: Number of failures

        Returns:
            Probability P(X <= k)
        """
        ...

    def ppf(self, q: float) -> float:
        """
        Percent point function (inverse CDF).

        Parameters:
            q: Probability value in [0, 1]

        Returns:
            k such that CDF(k) >= q
        """
        ...

    def rvs(self, size: int) -> List[float]:
        """
        Random variates.

        Parameters:
            size: Number of random samples

        Returns:
            List of random samples (number of failures before first success)
        """
        ...

# =============================================================================
# FFT Module
# =============================================================================

def fft_py(data: NDArray[np.float64]) -> Dict[str, List[float]]:
    """Compute FFT. Returns dict with real and imag arrays."""
    ...

def ifft_py(
    real: NDArray[np.float64],
    imag: NDArray[np.float64]
) -> Dict[str, List[float]]:
    """Compute inverse FFT. Returns dict with real and imag arrays."""
    ...

def rfft_py(data: NDArray[np.float64]) -> Dict[str, List[float]]:
    """Compute real FFT. Returns dict with real and imag arrays."""
    ...

def irfft_py(
    real: NDArray[np.float64],
    imag: NDArray[np.float64],
    n: int
) -> NDArray[np.float64]:
    """Compute inverse real FFT."""
    ...

def dct_py(data: NDArray[np.float64], dct_type: int = 2) -> List[float]:
    """Compute DCT (type 1, 2, 3, or 4)."""
    ...

def idct_py(data: NDArray[np.float64], dct_type: int = 2) -> List[float]:
    """Compute inverse DCT."""
    ...

def fftfreq_py(n: int, d: float = 1.0) -> List[float]:
    """Generate FFT sample frequencies."""
    ...

def rfftfreq_py(n: int, d: float = 1.0) -> List[float]:
    """Generate real FFT sample frequencies."""
    ...

def fftshift_py(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shift zero-frequency component to center."""
    ...

def ifftshift_py(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of fftshift."""
    ...

def next_fast_len_py(n: int, real: bool = False) -> int:
    """Find next fast FFT size."""
    ...

def fft2_py(
    data: NDArray[np.float64],
    s: Optional[Tuple[int, int]] = None,
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute the 2-D discrete Fourier Transform.

    Parameters
    ----------
    data : ndarray of shape (M, N)
        Input 2-D array
    s : tuple of (int, int), optional
        Shape of the output transform; if None, uses input shape

    Returns
    -------
    dict
        Dictionary with 'real' and 'imag' arrays of the 2-D FFT result
    """
    ...

def rfft2_py(
    data: NDArray[np.float64],
    s: Optional[Tuple[int, int]] = None,
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute the 2-D real discrete Fourier Transform.

    Parameters
    ----------
    data : ndarray of shape (M, N)
        Input 2-D real-valued array
    s : tuple of (int, int), optional
        Shape of the output transform; if None, uses input shape

    Returns
    -------
    dict
        Dictionary with 'real' and 'imag' arrays of the 2-D real FFT result
    """
    ...

def ifft2_py(
    real: NDArray[np.float64],
    imag: NDArray[np.float64],
    s: Optional[Tuple[int, int]] = None,
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute the 2-D inverse discrete Fourier Transform.

    Parameters
    ----------
    real : ndarray
        Real part of the input spectrum
    imag : ndarray
        Imaginary part of the input spectrum
    s : tuple of (int, int), optional
        Shape of the output; if None, uses input shape

    Returns
    -------
    dict
        Dictionary with 'real' and 'imag' arrays of the reconstructed signal
    """
    ...

def irfft2_py(
    real: NDArray[np.float64],
    imag: NDArray[np.float64],
    s: Optional[Tuple[int, int]] = None,
) -> NDArray[np.float64]:
    """
    Compute the 2-D inverse real discrete Fourier Transform.

    Parameters
    ----------
    real : ndarray
        Real part of the rfft2 output
    imag : ndarray
        Imaginary part of the rfft2 output
    s : tuple of (int, int), optional
        Shape of the output (required when output shape is ambiguous)

    Returns
    -------
    ndarray
        Real-valued 2-D reconstructed signal
    """
    ...

# =============================================================================
# Optimization Module
# =============================================================================

def minimize_py(
    fun: Any,
    x0: List[float],
    method: str = "bfgs",
    options: Optional[Dict[str, Any]] = None,
    bounds: Optional[List[tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Minimize a function of one or more variables.

    Parameters:
        fun: Objective function taking array-like and returning a float
        x0: Initial guess
        method: Optimization method
            - 'nelder-mead': Nelder-Mead simplex
            - 'powell': Powell's method
            - 'cg': Conjugate gradient
            - 'bfgs': BFGS quasi-Newton
            - 'lbfgs': Limited-memory BFGS
            - 'lbfgsb': L-BFGS-B with bounds
            - 'newton-cg': Newton-CG
            - 'trust-ncg': Newton trust-region
            - 'sr1': SR1 quasi-Newton
            - 'dfp': DFP quasi-Newton
        options: Dict with 'maxiter', 'ftol', 'gtol'
        bounds: Optional list of (min, max) bounds for each variable

    Returns:
        Dict with 'x' (solution), 'fun' (function value), 'success',
        'nit' (iterations), 'nfev' (function evaluations), 'message'
    """
    ...

def minimize_scalar_py(
    f: Any,
    bracket: tuple[float, float],
    method: str = "brent",
    maxiter: int = 100,
    tol: float = 1e-8
) -> Dict[str, Any]:
    """
    Minimize a scalar function.

    Parameters:
        f: Objective function taking a float and returning a float
        bracket: Initial bracket (a, b) where minimum is searched
        method: Optimization method ('brent', 'golden', 'bounded')
        maxiter: Maximum iterations
        tol: Tolerance for termination

    Returns:
        Dict with 'x' (minimum location), 'fun' (minimum value),
        'nfev' (function evaluations), 'success' (bool)
    """
    ...

def brentq_py(
    fun: Any,
    a: float,
    b: float,
    xtol: float = 1e-12,
    maxiter: int = 100
) -> Dict[str, Any]:
    """
    Find root of a scalar function using Brent's method.

    Parameters:
        fun: Function for which to find the root
        a: Lower bound of the bracket
        b: Upper bound of the bracket
        xtol: Absolute tolerance
        maxiter: Maximum iterations

    Returns:
        Dict with 'x' (root location), 'fun' (function value at root),
        'iterations', 'success'
    """
    ...

def differential_evolution_py(
    f: Any,
    bounds: List[tuple[float, float]],
    maxiter: int = 1000,
    popsize: int = 15,
    tol: float = 0.01,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Global optimization using differential evolution.

    Parameters:
        f: Objective function taking an array and returning a float
        bounds: List of (min, max) bounds for each variable
        maxiter: Maximum iterations
        popsize: Population size multiplier
        tol: Tolerance for convergence
        seed: Random seed for reproducibility

    Returns:
        Dict with 'x' (optimal point), 'fun' (optimal value),
        'nfev' (function evaluations), 'success' (bool)
    """
    ...

def curve_fit_py(
    f: Callable[..., float],
    xdata: List[float],
    ydata: List[float],
    p0: Optional[List[float]] = None,
    method: str = "lm",
    maxfev: int = 1000
) -> Dict[str, Any]:
    """
    Use non-linear least squares to fit a function to data.

    Similar to scipy.optimize.curve_fit.

    Parameters:
        f: Model function f(x, *params) that takes independent variable
           and parameters. Example: lambda x, a, b: a * exp(b * x)
        xdata: The independent variable where data is measured
        ydata: The dependent data to fit
        p0: Initial guess for parameters. If None, defaults to [1.0, 1.0]
        method: Optimization method - 'lm' (Levenberg-Marquardt),
                'trf' (Trust Region Reflective), or 'dogbox'
        maxfev: Maximum number of function evaluations

    Returns:
        Dictionary with:
        - popt: Optimized parameters as numpy array
        - success: Whether optimization succeeded
        - nfev: Number of function evaluations
        - message: Status message

    Examples:
        >>> def model(x, a, b):
        ...     return a * np.exp(b * x)
        >>> xdata = [0.0, 1.0, 2.0, 3.0]
        >>> ydata = [1.0, 2.7, 7.4, 20.1]
        >>> result = scirs2.curve_fit_py(model, xdata, ydata, p0=[1.0, 1.0])
        >>> print(result['popt'])  # Optimized [a, b]

        >>> # Fit quadratic: y = a*x^2 + b*x + c
        >>> def quadratic(x, a, b, c):
        ...     return a * x**2 + b * x + c
        >>> xdata = [0.0, 1.0, 2.0, 3.0, 4.0]
        >>> ydata = [1.0, 3.5, 7.0, 11.5, 17.0]
        >>> result = scirs2.curve_fit_py(quadratic, xdata, ydata, p0=[0.5, 2.0, 1.0])
    """
    ...

# =============================================================================
# Special Functions Module
# =============================================================================

# Gamma functions
def gamma_py(x: float) -> float:
    """Compute gamma function Γ(x)."""
    ...

def lgamma_py(x: float) -> float:
    """Compute log of gamma function ln(Γ(x))."""
    ...

def digamma_py(x: float) -> float:
    """Compute digamma (psi) function ψ(x) = Γ'(x)/Γ(x)."""
    ...

def beta_py(a: float, b: float) -> float:
    """Compute beta function B(a, b)."""
    ...

# Bessel functions
def j0_py(x: float) -> float:
    """Bessel function of the first kind, order 0."""
    ...

def j1_py(x: float) -> float:
    """Bessel function of the first kind, order 1."""
    ...

def jn_py(n: int, x: float) -> float:
    """Bessel function of the first kind, order n."""
    ...

def y0_py(x: float) -> float:
    """Bessel function of the second kind, order 0."""
    ...

def y1_py(x: float) -> float:
    """Bessel function of the second kind, order 1."""
    ...

def yn_py(n: int, x: float) -> float:
    """Bessel function of the second kind, order n."""
    ...

# Modified Bessel functions
def i0_py(x: float) -> float:
    """Modified Bessel function of the first kind, order 0."""
    ...

def i1_py(x: float) -> float:
    """Modified Bessel function of the first kind, order 1."""
    ...

def k0_py(x: float) -> float:
    """Modified Bessel function of the second kind, order 0."""
    ...

def k1_py(x: float) -> float:
    """Modified Bessel function of the second kind, order 1."""
    ...

# Error functions
def erf_py(x: float) -> float:
    """Compute error function erf(x)."""
    ...

def erfc_py(x: float) -> float:
    """Compute complementary error function erfc(x) = 1 - erf(x)."""
    ...

def erfinv_py(x: float) -> float:
    """Compute inverse error function."""
    ...

def erfcinv_py(x: float) -> float:
    """Compute inverse complementary error function."""
    ...

def erfcx_py(x: float) -> float:
    """Compute scaled complementary error function erfcx(x) = exp(x²) * erfc(x)."""
    ...

def erfi_py(x: float) -> float:
    """Compute imaginary error function erfi(x) = -i * erf(ix)."""
    ...

def dawsn_py(x: float) -> float:
    """Compute Dawson's integral F(x) = exp(-x²) * ∫₀ˣ exp(t²) dt."""
    ...

# Combinatorial functions
def factorial_py(n: int) -> float:
    """
    Compute factorial n!

    Parameters:
        n: Non-negative integer

    Returns:
        n! as a float
    """
    ...

def comb_py(n: int, k: int) -> float:
    """
    Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)

    Parameters:
        n: Total number of items
        k: Number of items to choose

    Returns:
        Binomial coefficient as a float
    """
    ...

def perm_py(n: int, k: int) -> float:
    """
    Compute permutations P(n, k) = n! / (n-k)!

    Parameters:
        n: Total number of items
        k: Number of items to arrange

    Returns:
        Number of permutations as a float
    """
    ...

# Elliptic integrals
def ellipk_py(m: float) -> float:
    """
    Complete elliptic integral of the first kind K(m).

    Parameters:
        m: Parameter (0 ≤ m ≤ 1)

    Returns:
        K(m) = ∫₀^(π/2) dθ / sqrt(1 - m*sin²(θ))
    """
    ...

def ellipe_py(m: float) -> float:
    """
    Complete elliptic integral of the second kind E(m).

    Parameters:
        m: Parameter (0 ≤ m ≤ 1)

    Returns:
        E(m) = ∫₀^(π/2) sqrt(1 - m*sin²(θ)) dθ
    """
    ...

def ellipkinc_py(phi: float, m: float) -> float:
    """
    Incomplete elliptic integral of the first kind F(φ, m).

    Parameters:
        phi: Amplitude
        m: Parameter (0 ≤ m ≤ 1)

    Returns:
        F(φ, m) = ∫₀^φ dθ / sqrt(1 - m*sin²(θ))
    """
    ...

def ellipeinc_py(phi: float, m: float) -> float:
    """
    Incomplete elliptic integral of the second kind E(φ, m).

    Parameters:
        phi: Amplitude
        m: Parameter (0 ≤ m ≤ 1)

    Returns:
        E(φ, m) = ∫₀^φ sqrt(1 - m*sin²(θ)) dθ
    """
    ...

# Vectorized versions
def gamma_array_py(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute gamma function for array of values."""
    ...

def erf_array_py(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute error function for array of values."""
    ...

def j0_array_py(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute J0 Bessel function for array of values."""
    ...

# =============================================================================
# Integration Module
# =============================================================================

def trapezoid_array_py(
    y: NDArray[np.float64],
    x: Optional[NDArray[np.float64]] = None,
    dx: float = 1.0
) -> float:
    """
    Integrate using trapezoidal rule.

    Parameters:
        y: Array of function values
        x: Array of sample points (optional)
        dx: Sample spacing if x is not provided
    """
    ...

def simpson_array_py(
    y: NDArray[np.float64],
    x: Optional[NDArray[np.float64]] = None,
    dx: float = 1.0
) -> float:
    """
    Integrate using Simpson's rule.

    Parameters:
        y: Array of function values
        x: Array of sample points (optional)
        dx: Sample spacing if x is not provided
    """
    ...

def cumulative_trapezoid_py(
    y: NDArray[np.float64],
    x: Optional[NDArray[np.float64]] = None,
    dx: float = 1.0,
    initial: Optional[float] = None
) -> NDArray[np.float64]:
    """
    Compute cumulative integral using trapezoidal rule.

    Parameters:
        y: Array of function values
        x: Array of sample points (optional)
        dx: Sample spacing if x is not provided
        initial: Initial value to prepend to result
    """
    ...

def romberg_array_py(
    y: NDArray[np.float64],
    dx: float = 1.0
) -> float:
    """
    Integrate using Romberg integration on sampled data.

    Parameters:
        y: Array of function values (length must be 2^k + 1)
        dx: Sample spacing
    """
    ...

def quad_py(
    fun: Any,
    a: float,
    b: float,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    maxiter: int = 500
) -> Dict[str, Any]:
    """
    Adaptive quadrature integration.

    Parameters:
        fun: Function to integrate
        a: Lower bound
        b: Upper bound
        epsabs: Absolute error tolerance
        epsrel: Relative error tolerance
        maxiter: Maximum function evaluations

    Returns:
        Dict with 'value' (integral), 'error' (estimated error),
        'neval' (function evaluations), 'success' (bool)
    """
    ...

def solve_ivp_py(
    fun: Any,
    t_span: tuple[float, float],
    y0: List[float],
    method: str = "RK45",
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: Optional[float] = None
) -> Dict[str, Any]:
    """
    Solve an initial value problem for a system of ODEs.

    Solves dy/dt = f(t, y) with y(t0) = y0.

    Parameters:
        fun: Function computing dy/dt = f(t, y)
        t_span: Integration interval (t0, tf)
        y0: Initial state
        method: ODE solver method
            - 'RK45': Explicit Runge-Kutta 4(5) (default)
            - 'RK23': Explicit Runge-Kutta 2(3)
            - 'DOP853': Explicit Runge-Kutta 8(5,3)
            - 'Radau': Implicit Runge-Kutta (stiff)
            - 'BDF': Backward differentiation formula (stiff)
            - 'LSODA': Adams/BDF with automatic stiffness detection
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_step: Maximum step size (optional)

    Returns:
        Dict with 't' (time points), 'y' (solutions as 2D array),
        'nfev' (function evaluations), 'success', 'message'
    """
    ...

# =============================================================================
# Interpolation Module
# =============================================================================

class Interp1d:
    """1D interpolation class."""

    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        method: str = "linear",
        extrapolate: str = "error"
    ) -> None:
        """
        Create 1D interpolator.

        Parameters:
            x: x coordinates (must be sorted)
            y: y coordinates
            method: 'linear', 'nearest', 'cubic', or 'pchip'
            extrapolate: 'error', 'nearest', or 'extrapolate'
        """
        ...

    def __call__(self, x_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate interpolator at new points."""
        ...

    def eval_single(self, x: float) -> float:
        """Evaluate interpolator at a single point."""
        ...

def interp_py(
    x: NDArray[np.float64],
    xp: NDArray[np.float64],
    fp: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Linear interpolation (similar to numpy.interp).

    Parameters:
        x: x-coordinates at which to evaluate
        xp: x-coordinates of data points
        fp: y-coordinates of data points
    """
    ...

def interp_with_bounds_py(
    x: NDArray[np.float64],
    xp: NDArray[np.float64],
    fp: NDArray[np.float64],
    left: Optional[float] = None,
    right: Optional[float] = None
) -> NDArray[np.float64]:
    """
    Linear interpolation with boundary handling.

    Parameters:
        x: x-coordinates at which to evaluate
        xp: x-coordinates of data points
        fp: y-coordinates of data points
        left: Value to return for x < xp[0]
        right: Value to return for x > xp[-1]
    """
    ...

class CubicSpline:
    """
    Cubic spline interpolation class.

    Provides C2-continuous (continuous function, first, and second derivatives)
    interpolation through data points using piecewise cubic polynomials.

    Examples:
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> y = np.array([0.0, 1.0, 4.0, 9.0])
        >>> spline = scirs2.CubicSpline(x, y)
        >>>
        >>> # Evaluate at new points
        >>> x_new = np.array([0.5, 1.5, 2.5])
        >>> y_new = spline(x_new)
        >>>
        >>> # Compute derivatives
        >>> dy_dx = spline.derivative(x_new, nu=1)  # First derivative
        >>> d2y_dx2 = spline.derivative(x_new, nu=2)  # Second derivative
        >>>
        >>> # Integrate
        >>> integral = spline.integrate(0.0, 3.0)
    """

    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        bc_type: str = "natural"
    ) -> None:
        """
        Create a cubic spline interpolator.

        Parameters:
            x: x coordinates (must be strictly increasing)
            y: y coordinates
            bc_type: Boundary condition type - 'natural', 'not-a-knot', or 'periodic'
                    'natural': Zero second derivative at endpoints (default)
                    'not-a-knot': Maximum smoothness at second and second-to-last points
                    'periodic': Function and derivatives match at endpoints

        Raises:
            ValueError: If x is not strictly increasing or if x and y have different lengths
        """
        ...

    def __call__(self, x_new: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the spline at new points.

        Parameters:
            x_new: Points at which to evaluate the spline

        Returns:
            Interpolated values at x_new
        """
        ...

    def eval_single(self, x: float) -> float:
        """
        Evaluate the spline at a single point.

        Parameters:
            x: Point at which to evaluate

        Returns:
            Interpolated value at x
        """
        ...

    def derivative(
        self,
        x_new: NDArray[np.float64],
        nu: int = 1
    ) -> NDArray[np.float64]:
        """
        Compute derivatives of the spline.

        Parameters:
            x_new: Points at which to evaluate derivatives
            nu: Derivative order (1 for first derivative, 2 for second, etc.)

        Returns:
            Derivative values at x_new

        Examples:
            >>> spline = scirs2.CubicSpline(x, y)
            >>> # First derivative
            >>> dy_dx = spline.derivative(x_new, nu=1)
            >>> # Second derivative
            >>> d2y_dx2 = spline.derivative(x_new, nu=2)
        """
        ...

    def integrate(self, a: float, b: float) -> float:
        """
        Integrate the spline over an interval.

        Parameters:
            a: Lower bound of integration
            b: Upper bound of integration

        Returns:
            Definite integral from a to b

        Examples:
            >>> spline = scirs2.CubicSpline(x, y)
            >>> area = spline.integrate(0.0, 3.0)
        """
        ...

class Interp2d:
    """
    2D interpolation on regular grids.

    Provides bivariate interpolation for data defined on a 2D rectangular grid.
    Similar to scipy.interpolate.interp2d but for regular grids only.

    Examples:
        >>> # Create a 2D grid
        >>> x = np.array([0.0, 1.0, 2.0])
        >>> y = np.array([0.0, 1.0])
        >>> # z[i, j] corresponds to point (x[j], y[i])
        >>> z = np.array([[0.0, 1.0, 2.0],
        ...               [1.0, 2.0, 3.0]])
        >>>
        >>> interp = scirs2.Interp2d(x, y, z, kind="linear")
        >>>
        >>> # Evaluate at a single point
        >>> value = interp(0.5, 0.5)
        >>>
        >>> # Evaluate at multiple points
        >>> x_new = np.array([0.5, 1.0, 1.5])
        >>> y_new = np.array([0.5, 0.5, 0.5])
        >>> values = interp.eval_array(x_new, y_new)
        >>>
        >>> # Evaluate on a regular grid
        >>> x_grid = np.array([0.0, 0.5, 1.0])
        >>> y_grid = np.array([0.0, 0.5, 1.0])
        >>> z_grid = interp.eval_grid(x_grid, y_grid)  # Shape: (3, 3)
    """

    def __init__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        kind: str = "linear"
    ) -> None:
        """
        Create a 2D interpolator on a regular grid.

        Parameters:
            x: x coordinates (1D array, must be sorted), length n_x
            y: y coordinates (1D array, must be sorted), length n_y
            z: z values on the grid (2D array with shape (n_y, n_x))
               Note: z[i, j] corresponds to point (x[j], y[i])
            kind: Interpolation method - 'linear', 'cubic', or 'quintic'
                  'linear': Bilinear interpolation (fast, C0 continuous)
                  'cubic': Bicubic interpolation (smooth, C1 continuous)
                  'quintic': Biquintic interpolation (very smooth, requires >= 6 points)

        Raises:
            ValueError: If z.shape != (len(y), len(x)) or if x/y are not sorted

        Notes:
            The z array indexing follows the convention: z[row, col] = z[y_index, x_index]
            This matches NumPy's meshgrid convention with indexing='ij'.
        """
        ...

    def __call__(self, x: float, y: float) -> float:
        """
        Evaluate the interpolator at a single point.

        Parameters:
            x: x coordinate
            y: y coordinate

        Returns:
            Interpolated value at (x, y)
        """
        ...

    def eval_array(
        self,
        x_new: NDArray[np.float64],
        y_new: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Evaluate at multiple scattered points.

        Parameters:
            x_new: x coordinates (must have same length as y_new)
            y_new: y coordinates (must have same length as x_new)

        Returns:
            1D array of interpolated values, same length as x_new/y_new

        Examples:
            >>> x_points = np.array([0.5, 1.0, 1.5])
            >>> y_points = np.array([0.5, 0.7, 0.3])
            >>> values = interp.eval_array(x_points, y_points)
        """
        ...

    def eval_grid(
        self,
        x_new: NDArray[np.float64],
        y_new: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Evaluate on a regular grid (Cartesian product of x_new and y_new).

        Parameters:
            x_new: x coordinates for output grid (1D array)
            y_new: y coordinates for output grid (1D array)

        Returns:
            2D array with shape (len(y_new), len(x_new))
            Result[i, j] corresponds to point (x_new[j], y_new[i])

        Examples:
            >>> x_grid = np.linspace(0, 2, 5)
            >>> y_grid = np.linspace(0, 1, 3)
            >>> z_grid = interp.eval_grid(x_grid, y_grid)
            >>> # z_grid.shape = (3, 5)
        """
        ...

# =============================================================================
# Signal Processing Module
# =============================================================================

def convolve_py(
    a: NDArray[np.float64],
    v: NDArray[np.float64],
    mode: str = "full"
) -> NDArray[np.float64]:
    """
    Convolve two 1-D arrays.

    Parameters:
        a: First input array
        v: Second input array
        mode: 'full', 'same', or 'valid'
    """
    ...

def correlate_py(
    a: NDArray[np.float64],
    v: NDArray[np.float64],
    mode: str = "full"
) -> NDArray[np.float64]:
    """
    Cross-correlation of two 1-D arrays.

    Parameters:
        a: First input array
        v: Second input array
        mode: 'full', 'same', or 'valid'
    """
    ...

def hilbert_py(x: NDArray[np.float64]) -> Dict[str, NDArray[np.float64]]:
    """
    Compute analytic signal using Hilbert transform.

    Returns:
        Dict with 'real' and 'imag' arrays
    """
    ...

# Window functions
def hann_py(n: int) -> NDArray[np.float64]:
    """Generate Hann window of length n."""
    ...

def hamming_py(n: int) -> NDArray[np.float64]:
    """Generate Hamming window of length n."""
    ...

def blackman_py(n: int) -> NDArray[np.float64]:
    """Generate Blackman window of length n."""
    ...

def bartlett_py(n: int) -> NDArray[np.float64]:
    """Generate Bartlett (triangular) window of length n."""
    ...

def kaiser_py(n: int, beta: float) -> NDArray[np.float64]:
    """
    Generate Kaiser window.

    Parameters:
        n: Window length
        beta: Shape parameter (0=rectangular, 5=Hamming-like, 6=Hann-like)
    """
    ...

def find_peaks_py(
    x: NDArray[np.float64],
    height: Optional[float] = None,
    distance: Optional[int] = None
) -> NDArray[np.int64]:
    """
    Find peaks in a 1-D array.

    Parameters:
        x: Input array
        height: Minimum peak height
        distance: Minimum distance between peaks

    Returns:
        Array of peak indices
    """
    ...

# Filter design functions
def butter_py(
    order: int,
    cutoff: float,
    filter_type: str = "lowpass"
) -> Dict[str, NDArray[np.float64]]:
    """
    Design a Butterworth digital filter.

    Parameters:
        order: Filter order
        cutoff: Cutoff frequency (normalized 0-1, where 1 is Nyquist)
        filter_type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'

    Returns:
        Dict with 'b' (numerator) and 'a' (denominator) coefficients
    """
    ...

def cheby1_py(
    order: int,
    ripple: float,
    cutoff: float,
    filter_type: str = "lowpass"
) -> Dict[str, NDArray[np.float64]]:
    """
    Design a Chebyshev Type I digital filter.

    Parameters:
        order: Filter order
        ripple: Passband ripple in dB
        cutoff: Cutoff frequency (normalized 0-1, where 1 is Nyquist)
        filter_type: 'lowpass' or 'highpass'

    Returns:
        Dict with 'b' (numerator) and 'a' (denominator) coefficients
    """
    ...

def firwin_py(
    numtaps: int,
    cutoff: float,
    window: str = "hamming",
    pass_zero: bool = True
) -> NDArray[np.float64]:
    """
    Design a FIR filter using window method.

    Parameters:
        numtaps: Number of filter taps (filter order + 1)
        cutoff: Cutoff frequency (normalized 0-1, where 1 is Nyquist)
        window: Window function ('hamming', 'hann', 'blackman', 'kaiser')
        pass_zero: If True, lowpass; if False, highpass

    Returns:
        Filter coefficients as numpy array
    """
    ...

# =============================================================================
# Spatial Module
# =============================================================================

# Distance functions
def euclidean_py(
    u: NDArray[np.float64],
    v: NDArray[np.float64]
) -> float:
    """Euclidean distance between two points."""
    ...

def cityblock_py(
    u: NDArray[np.float64],
    v: NDArray[np.float64]
) -> float:
    """Manhattan (city block) distance between two points."""
    ...

def chebyshev_py(
    u: NDArray[np.float64],
    v: NDArray[np.float64]
) -> float:
    """Chebyshev distance between two points."""
    ...

def minkowski_py(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    p: float
) -> float:
    """
    Minkowski distance between two points.

    Parameters:
        u, v: Input arrays
        p: Order of the norm (p=1 is cityblock, p=2 is euclidean)
    """
    ...

def cosine_py(
    u: NDArray[np.float64],
    v: NDArray[np.float64]
) -> float:
    """Cosine distance between two points (1 - cosine similarity)."""
    ...

def spatial_hamming_distance_py(
    u: NDArray[np.float64],
    v: NDArray[np.float64]
) -> float:
    """
    Hamming distance between two points (proportion of differing coordinates).

    Matches scipy.spatial.distance.hamming. Named distinctly from hamming_py
    (the signal-processing Hamming *window* function) and from
    hamming_distance_py (reserved for the text module's string/character
    Hamming distance) to avoid naming confusion between the three.
    """
    ...

def pdist_py(
    x: NDArray[np.float64],
    metric: str = "euclidean"
) -> NDArray[np.float64]:
    """
    Compute pairwise distances between observations.

    Parameters:
        x: Input array of shape (n, m)
        metric: 'euclidean', 'cityblock', 'manhattan', 'chebyshev', or 'hamming'

    Returns:
        Condensed distance matrix (n*(n-1)/2,)
    """
    ...

def cdist_py(
    xa: NDArray[np.float64],
    xb: NDArray[np.float64],
    metric: str = "euclidean"
) -> NDArray[np.float64]:
    """
    Compute distances between each pair from two sets of observations.

    Parameters:
        xa: Input array of shape (na, m)
        xb: Input array of shape (nb, m)
        metric: 'euclidean', 'cityblock', 'manhattan', 'chebyshev', or 'hamming'

    Returns:
        Distance matrix of shape (na, nb)
    """
    ...

def squareform_py(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert condensed distance matrix to square form.

    Parameters:
        x: Condensed distance matrix from pdist

    Returns:
        Square distance matrix
    """
    ...

# Convex Hull
def convex_hull_py(points: NDArray[np.float64]) -> Dict[str, Any]:
    """
    Compute the convex hull of a set of points.

    Parameters:
        points: Array of shape (n, k) containing n points in k dimensions

    Returns:
        Dict with 'vertices' (indices), 'simplices', 'volume', and 'area'
    """
    ...

class ConvexHullPy:
    """ConvexHull class for working with convex hulls."""

    def __init__(self, points: NDArray[np.float64]) -> None:
        """
        Create a ConvexHull from points.

        Parameters:
            points: Array of shape (n, k) containing n points in k dimensions
        """
        ...

    def vertices(self) -> NDArray[np.int64]:
        """Get the indices of vertices that form the convex hull."""
        ...

    def simplices(self) -> List[List[int]]:
        """Get the simplices (facets) of the convex hull."""
        ...

    def volume(self) -> float:
        """Calculate the volume of the convex hull."""
        ...

    def area(self) -> float:
        """Calculate the surface area of the convex hull."""
        ...

    def contains(self, point: NDArray[np.float64]) -> bool:
        """Check if a point is inside the convex hull."""
        ...

# KD-Tree spatial data structure
class KDTree:
    """KD-Tree for efficient nearest neighbor searches."""

    def __init__(self, data: NDArray[np.float64]) -> None:
        """
        Construct a KD-Tree from points.

        Parameters:
            data: Array of shape (n, k) containing n points in k dimensions
        """
        ...

    def query(
        self,
        point: NDArray[np.float64],
        k: int = 1
    ) -> Dict[str, NDArray]:
        """
        Query the tree for k nearest neighbors.

        Parameters:
            point: Query point
            k: Number of nearest neighbors to find

        Returns:
            Dict with 'indices' and 'distances' arrays
        """
        ...

    def query_radius(
        self,
        point: NDArray[np.float64],
        r: float
    ) -> Dict[str, NDArray]:
        """
        Query the tree for all points within radius r.

        Parameters:
            point: Query point
            r: Search radius

        Returns:
            Dict with 'indices' and 'distances' arrays
        """
        ...

# =============================================================================
# Sparse Matrix Module
# =============================================================================

def csr_array_from_triplets(
    rows: List[int],
    cols: List[int],
    data: List[float],
    shape: Tuple[int, int],
    sum_duplicates: bool = False,
) -> Dict[str, Any]:
    """
    Create a CSR (Compressed Sparse Row) sparse array from triplets.

    Parameters
    ----------
    rows : list of int
        Row indices of non-zero elements
    cols : list of int
        Column indices of non-zero elements
    data : list of float
        Non-zero values
    shape : tuple of (int, int)
        Shape of the resulting matrix (nrows, ncols)
    sum_duplicates : bool, optional
        If True, sum values at duplicate (row, col) positions (default: False)

    Returns
    -------
    dict
        Dictionary with keys: 'format', 'shape', 'nnz', 'indptr', 'indices', 'data'
    """
    ...

def coo_array_from_triplets(
    rows: List[int],
    cols: List[int],
    data: List[float],
    shape: Tuple[int, int],
    sum_duplicates: bool = False,
) -> Dict[str, Any]:
    """
    Create a COO (Coordinate) sparse array from triplets.

    Parameters
    ----------
    rows : list of int
        Row indices of non-zero elements
    cols : list of int
        Column indices of non-zero elements
    data : list of float
        Non-zero values
    shape : tuple of (int, int)
        Shape of the resulting matrix (nrows, ncols)
    sum_duplicates : bool, optional
        If True, sum values at duplicate (row, col) positions (default: False)

    Returns
    -------
    dict
        Dictionary with keys: 'format', 'shape', 'nnz', 'row', 'col', 'data'
    """
    ...

def csc_array_from_triplets(
    rows: List[int],
    cols: List[int],
    data: List[float],
    shape: Tuple[int, int],
    sum_duplicates: bool = False,
) -> Dict[str, Any]:
    """
    Create a CSC (Compressed Sparse Column) sparse array from triplets.

    Parameters
    ----------
    rows : list of int
        Row indices of non-zero elements
    cols : list of int
        Column indices of non-zero elements
    data : list of float
        Non-zero values
    shape : tuple of (int, int)
        Shape of the resulting matrix (nrows, ncols)
    sum_duplicates : bool, optional
        If True, sum values at duplicate (row, col) positions (default: False)

    Returns
    -------
    dict
        Dictionary with keys: 'format', 'shape', 'nnz', 'indptr', 'indices', 'data'
    """
    ...

def sparse_eye_py(n: int) -> Dict[str, Any]:
    """
    Create a sparse identity matrix of size n x n.

    Parameters
    ----------
    n : int
        Size of the square identity matrix

    Returns
    -------
    dict
        CSR sparse matrix representation of the identity matrix
    """
    ...

def sparse_diag_py(diag: NDArray[np.float64]) -> Dict[str, Any]:
    """
    Create a sparse diagonal matrix from a 1-D array.

    Parameters
    ----------
    diag : ndarray of shape (n,)
        Diagonal elements

    Returns
    -------
    dict
        CSR sparse matrix with the given diagonal
    """
    ...

def sparse_to_dense_py(sparse_dict: Dict[str, Any]) -> NDArray[np.float64]:
    """
    Convert a sparse matrix (dict representation) to a dense numpy array.

    Parameters
    ----------
    sparse_dict : dict
        Sparse matrix dict as returned by csr_array_from_triplets or similar

    Returns
    -------
    ndarray
        Dense 2-D array
    """
    ...

def sparse_matvec_py(
    sparse_dict: Dict[str, Any],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Multiply a sparse matrix by a dense vector: y = A @ x.

    Parameters
    ----------
    sparse_dict : dict
        Sparse matrix dict (CSR format)
    x : ndarray of shape (n,)
        Dense vector

    Returns
    -------
    ndarray of shape (m,)
        Result of matrix-vector multiplication
    """
    ...

def sparse_matmul_py(
    a_dict: Dict[str, Any],
    b_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Multiply two sparse matrices: C = A @ B.

    Parameters
    ----------
    a_dict : dict
        Left sparse matrix (CSR format)
    b_dict : dict
        Right sparse matrix (CSR format)

    Returns
    -------
    dict
        Resulting sparse matrix (CSR format)
    """
    ...

def sparse_transpose_py(sparse_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transpose a sparse matrix.

    Parameters
    ----------
    sparse_dict : dict
        Sparse matrix dict

    Returns
    -------
    dict
        Transposed sparse matrix
    """
    ...

# =============================================================================
# N-Dimensional Image Processing Module (ndimage)
# =============================================================================

def gaussian_filter_py(
    image: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """
    Apply Gaussian filter to an image.

    Parameters
    ----------
    image : ndarray
        Input image array (2-D or N-D)
    sigma : float
        Standard deviation of the Gaussian kernel

    Returns
    -------
    ndarray
        Filtered image
    """
    ...

def median_filter_py(
    image: NDArray[np.float64],
    size: int,
) -> NDArray[np.float64]:
    """
    Apply median filter to an image.

    Parameters
    ----------
    image : ndarray
        Input image array (2-D or N-D)
    size : int
        Size of the median filter kernel

    Returns
    -------
    ndarray
        Filtered image
    """
    ...

def uniform_filter_py(
    image: NDArray[np.float64],
    size: int,
) -> NDArray[np.float64]:
    """
    Apply uniform (box) filter to an image.

    Parameters
    ----------
    image : ndarray
        Input image array
    size : int
        Size of the filter kernel

    Returns
    -------
    ndarray
        Filtered image
    """
    ...

def sobel_py(
    image: NDArray[np.float64],
    axis: int,
) -> NDArray[np.float64]:
    """
    Apply Sobel edge-detection filter along a given axis.

    Parameters
    ----------
    image : ndarray
        Input image array
    axis : int
        Axis along which to apply the Sobel filter (0 for vertical, 1 for horizontal)

    Returns
    -------
    ndarray
        Edge-filtered image
    """
    ...

def laplace_py(image: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply Laplace filter for edge detection.

    Parameters
    ----------
    image : ndarray
        Input image array

    Returns
    -------
    ndarray
        Edge-filtered image using the Laplacian operator
    """
    ...

def bilateral_filter_py(
    image: NDArray[np.float64],
    sigma_spatial: float,
    sigma_intensity: float,
) -> NDArray[np.float64]:
    """
    Apply bilateral filter for edge-preserving smoothing.

    Parameters
    ----------
    image : ndarray
        Input image array
    sigma_spatial : float
        Standard deviation in the spatial domain
    sigma_intensity : float
        Standard deviation in the intensity domain

    Returns
    -------
    ndarray
        Filtered image
    """
    ...

def maximum_filter_py(
    image: NDArray[np.float64],
    size: int,
) -> NDArray[np.float64]:
    """
    Apply maximum filter to an image.

    Parameters
    ----------
    image : ndarray
        Input image array
    size : int
        Size of the filter kernel

    Returns
    -------
    ndarray
        Image with maximum filter applied
    """
    ...

def minimum_filter_py(
    image: NDArray[np.float64],
    size: int,
) -> NDArray[np.float64]:
    """
    Apply minimum filter to an image.

    Parameters
    ----------
    image : ndarray
        Input image array
    size : int
        Size of the filter kernel

    Returns
    -------
    ndarray
        Image with minimum filter applied
    """
    ...

def binary_erosion_py(
    image: NDArray[np.float64],
    iterations: int,
) -> NDArray[np.float64]:
    """
    Apply binary erosion morphological operation.

    Parameters
    ----------
    image : ndarray
        Binary input image (values 0 or 1)
    iterations : int
        Number of times to apply the erosion

    Returns
    -------
    ndarray
        Eroded binary image
    """
    ...

def binary_dilation_py(
    image: NDArray[np.float64],
    iterations: int,
) -> NDArray[np.float64]:
    """
    Apply binary dilation morphological operation.

    Parameters
    ----------
    image : ndarray
        Binary input image (values 0 or 1)
    iterations : int
        Number of times to apply the dilation

    Returns
    -------
    ndarray
        Dilated binary image
    """
    ...

def binary_opening_py(
    image: NDArray[np.float64],
    iterations: int,
) -> NDArray[np.float64]:
    """
    Apply binary opening morphological operation (erosion then dilation).

    Parameters
    ----------
    image : ndarray
        Binary input image (values 0 or 1)
    iterations : int
        Number of times to apply the operation

    Returns
    -------
    ndarray
        Opened binary image
    """
    ...

def binary_closing_py(
    image: NDArray[np.float64],
    iterations: int,
) -> NDArray[np.float64]:
    """
    Apply binary closing morphological operation (dilation then erosion).

    Parameters
    ----------
    image : ndarray
        Binary input image (values 0 or 1)
    iterations : int
        Number of times to apply the operation

    Returns
    -------
    ndarray
        Closed binary image
    """
    ...

def grey_erosion_py(
    image: NDArray[np.float64],
    size: int,
) -> NDArray[np.float64]:
    """
    Apply grey-scale erosion morphological operation.

    Parameters
    ----------
    image : ndarray
        Greyscale input image
    size : int
        Size of the structuring element

    Returns
    -------
    ndarray
        Eroded image
    """
    ...

def grey_dilation_py(
    image: NDArray[np.float64],
    size: int,
) -> NDArray[np.float64]:
    """
    Apply grey-scale dilation morphological operation.

    Parameters
    ----------
    image : ndarray
        Greyscale input image
    size : int
        Size of the structuring element

    Returns
    -------
    ndarray
        Dilated image
    """
    ...

def label_py(image: NDArray[np.float64]) -> Tuple[NDArray[np.int64], int]:
    """
    Label connected components in an image.

    Parameters
    ----------
    image : ndarray
        Binary input image

    Returns
    -------
    tuple of (ndarray, int)
        Labeled image array and number of labels found
    """
    ...

def distance_transform_edt_py(
    image: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute exact Euclidean distance transform.

    Parameters
    ----------
    image : ndarray
        Binary input image (0 = object, 1 = background)

    Returns
    -------
    ndarray
        Distance transform image where each pixel holds the distance to
        the nearest non-zero pixel
    """
    ...

def rotate_py(
    image: NDArray[np.float64],
    angle: float,
    reshape: bool = True,
) -> NDArray[np.float64]:
    """
    Rotate an image by a given angle.

    Parameters
    ----------
    image : ndarray
        Input image array
    angle : float
        Rotation angle in degrees (counter-clockwise)
    reshape : bool, optional
        If True, resize output to contain entire rotated image (default: True)

    Returns
    -------
    ndarray
        Rotated image
    """
    ...

def zoom_py(
    image: NDArray[np.float64],
    zoom: float,
) -> NDArray[np.float64]:
    """
    Zoom an image by a given factor.

    Parameters
    ----------
    image : ndarray
        Input image array
    zoom : float
        Zoom factor (> 1 = zoom in, < 1 = zoom out)

    Returns
    -------
    ndarray
        Zoomed image
    """
    ...

def shift_py(
    image: NDArray[np.float64],
    shift: List[float],
) -> NDArray[np.float64]:
    """
    Shift an image by given offsets.

    Parameters
    ----------
    image : ndarray
        Input image array
    shift : list of float
        Shift in pixels for each dimension

    Returns
    -------
    ndarray
        Shifted image
    """
    ...

def center_of_mass_py(image: NDArray[np.float64]) -> List[float]:
    """
    Compute the center of mass of an image.

    Parameters
    ----------
    image : ndarray
        Input image (treated as a mass distribution)

    Returns
    -------
    list of float
        Coordinates of the center of mass for each dimension
    """
    ...

def moments_py(image: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """
    Compute image moments up to a given order.

    Parameters
    ----------
    image : ndarray of shape (H, W)
        Input 2-D image
    order : int
        Maximum moment order to compute

    Returns
    -------
    ndarray
        Moment values
    """
    ...

def watershed_py(
    image: NDArray[np.float64],
    markers: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Apply watershed segmentation algorithm.

    Parameters
    ----------
    image : ndarray
        Greyscale image to segment
    markers : ndarray of int
        Marker array indicating seed regions (positive integers)

    Returns
    -------
    ndarray of int
        Labeled image with watershed segments
    """
    ...

def otsu_threshold_py(image: NDArray[np.float64]) -> float:
    """
    Compute Otsu's optimal threshold for image binarization.

    Parameters
    ----------
    image : ndarray
        Greyscale image

    Returns
    -------
    float
        Optimal threshold value that minimizes intra-class variance
    """
    ...

def threshold_binary_py(
    image: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """
    Apply a binary threshold to an image.

    Parameters
    ----------
    image : ndarray
        Input greyscale image
    threshold : float
        Threshold value; pixels above become 1.0, below become 0.0

    Returns
    -------
    ndarray
        Binary image
    """
    ...

def canny_py(
    image: NDArray[np.float64],
    low_threshold: float,
    high_threshold: float,
) -> NDArray[np.float64]:
    """
    Apply Canny edge detection.

    Parameters
    ----------
    image : ndarray
        Input greyscale image
    low_threshold : float
        Lower hysteresis threshold
    high_threshold : float
        Upper hysteresis threshold

    Returns
    -------
    ndarray
        Binary edge image
    """
    ...

def harris_corners_py(
    image: NDArray[np.float64],
    k: float,
) -> NDArray[np.float64]:
    """
    Detect Harris corners in an image.

    Parameters
    ----------
    image : ndarray
        Input greyscale image
    k : float
        Harris detector free parameter (typically 0.04-0.06)

    Returns
    -------
    ndarray
        Corner response map
    """
    ...

def psnr_py(
    original: NDArray[np.float64],
    compressed: NDArray[np.float64],
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Parameters
    ----------
    original : ndarray
        Original reference image
    compressed : ndarray
        Compared/compressed image

    Returns
    -------
    float
        PSNR in dB (higher is better)
    """
    ...

def ssim_py(
    img1: NDArray[np.float64],
    img2: NDArray[np.float64],
) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM).

    Parameters
    ----------
    img1 : ndarray
        First image
    img2 : ndarray
        Second image

    Returns
    -------
    float
        SSIM value in [-1, 1] (1.0 = identical images)
    """
    ...

def image_entropy_py(image: NDArray[np.float64]) -> float:
    """
    Compute the entropy of an image (measure of information content).

    Parameters
    ----------
    image : ndarray
        Input image

    Returns
    -------
    float
        Shannon entropy of the image's pixel intensity histogram
    """
    ...

# =============================================================================
# Graph Algorithms Module
# =============================================================================

def graph_from_edges_py(
    edges: List[Tuple[int, int]],
    weights: Optional[List[float]] = None,
) -> Any:
    """
    Create an undirected graph from edge list.

    Parameters
    ----------
    edges : list of (int, int)
        Edge list as pairs of node indices
    weights : list of float, optional
        Edge weights; if None, all weights default to 1.0

    Returns
    -------
    Graph object (opaque PyAny)
    """
    ...

def digraph_from_edges_py(
    edges: List[Tuple[int, int]],
    weights: Optional[List[float]] = None,
) -> Any:
    """
    Create a directed graph from edge list.

    Parameters
    ----------
    edges : list of (int, int)
        Edge list as pairs of (from, to) node indices
    weights : list of float, optional
        Edge weights; if None, all weights default to 1.0

    Returns
    -------
    DiGraph object (opaque PyAny)
    """
    ...

def erdos_renyi_graph_py(n: int, p: float) -> Any:
    """
    Generate an Erdos-Renyi random graph G(n, p).

    Parameters
    ----------
    n : int
        Number of nodes
    p : float
        Probability of edge between any pair of nodes

    Returns
    -------
    Graph object
    """
    ...

def barabasi_albert_graph_py(n: int, m: int) -> Any:
    """
    Generate a Barabasi-Albert scale-free random graph.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes

    Returns
    -------
    Graph object
    """
    ...

def watts_strogatz_graph_py(n: int, k: int, p: float) -> Any:
    """
    Generate a Watts-Strogatz small-world random graph.

    Parameters
    ----------
    n : int
        Number of nodes
    k : int
        Each node is joined with k nearest neighbors in ring topology
    p : float
        Probability of rewiring each edge

    Returns
    -------
    Graph object
    """
    ...

def complete_graph_py(n: int) -> Any:
    """
    Generate a complete graph with n nodes (every node connected to every other).

    Parameters
    ----------
    n : int
        Number of nodes

    Returns
    -------
    Graph object
    """
    ...

def path_graph_py(n: int) -> Any:
    """
    Generate a path graph with n nodes.

    Parameters
    ----------
    n : int
        Number of nodes in the path

    Returns
    -------
    Graph object
    """
    ...

def cycle_graph_py(n: int) -> Any:
    """
    Generate a cycle graph with n nodes.

    Parameters
    ----------
    n : int
        Number of nodes in the cycle

    Returns
    -------
    Graph object
    """
    ...

def star_graph_py(n: int) -> Any:
    """
    Generate a star graph with n leaf nodes (n+1 nodes total).

    Parameters
    ----------
    n : int
        Number of leaf nodes (center node is added automatically)

    Returns
    -------
    Graph object
    """
    ...

def bfs_py(graph: Any, start: int) -> List[int]:
    """
    Breadth-first search traversal order.

    Parameters
    ----------
    graph : Graph object
        Graph to traverse
    start : int
        Starting node index

    Returns
    -------
    list of int
        Nodes visited in BFS order
    """
    ...

def dfs_py(graph: Any, start: int) -> List[int]:
    """
    Depth-first search traversal order.

    Parameters
    ----------
    graph : Graph object
        Graph to traverse
    start : int
        Starting node index

    Returns
    -------
    list of int
        Nodes visited in DFS order
    """
    ...

def dijkstra_py(
    graph: Any,
    start: int,
) -> Dict[str, Any]:
    """
    Dijkstra's shortest path algorithm.

    Parameters
    ----------
    graph : Graph object
        Weighted graph
    start : int
        Source node index

    Returns
    -------
    dict
        Dictionary with 'distances' and 'predecessors' for each node
    """
    ...

def floyd_warshall_py(graph: Any) -> NDArray[np.float64]:
    """
    Floyd-Warshall all-pairs shortest paths algorithm.

    Parameters
    ----------
    graph : Graph object
        Weighted graph

    Returns
    -------
    ndarray of shape (n, n)
        Distance matrix where entry [i, j] is the shortest path from i to j.
        Infinity indicates no path.
    """
    ...

def connected_components_py(graph: Any) -> List[List[int]]:
    """
    Find all connected components in an undirected graph.

    Parameters
    ----------
    graph : Graph object
        Undirected graph

    Returns
    -------
    list of list of int
        Each inner list is a set of node indices forming one connected component
    """
    ...

def strongly_connected_components_py(graph: Any) -> List[List[int]]:
    """
    Find strongly connected components in a directed graph (Tarjan's algorithm).

    Parameters
    ----------
    graph : DiGraph object
        Directed graph

    Returns
    -------
    list of list of int
        Each inner list is a strongly connected component
    """
    ...

def articulation_points_py(graph: Any) -> List[int]:
    """
    Find articulation points (cut vertices) in an undirected graph.

    Articulation points are nodes whose removal increases the number of
    connected components.

    Parameters
    ----------
    graph : Graph object
        Undirected graph

    Returns
    -------
    list of int
        Node indices that are articulation points
    """
    ...

def bridges_py(graph: Any) -> List[Tuple[int, int]]:
    """
    Find bridges (cut edges) in an undirected graph.

    Bridges are edges whose removal increases the number of connected components.

    Parameters
    ----------
    graph : Graph object
        Undirected graph

    Returns
    -------
    list of (int, int)
        Edge pairs that are bridges
    """
    ...

def is_bipartite_py(graph: Any) -> bool:
    """
    Check if a graph is bipartite.

    Parameters
    ----------
    graph : Graph object
        Undirected graph

    Returns
    -------
    bool
        True if the graph is bipartite, False otherwise
    """
    ...

def betweenness_centrality_py(graph: Any) -> Dict[int, float]:
    """
    Compute betweenness centrality for all nodes.

    Betweenness centrality quantifies the number of times a node acts as a
    bridge along the shortest path between two other nodes.

    Parameters
    ----------
    graph : Graph object
        Graph (directed or undirected)

    Returns
    -------
    dict mapping int to float
        Betweenness centrality score for each node
    """
    ...

def closeness_centrality_py(graph: Any) -> Dict[int, float]:
    """
    Compute closeness centrality for all nodes.

    Closeness centrality measures how close a node is to all other nodes.

    Parameters
    ----------
    graph : Graph object
        Graph

    Returns
    -------
    dict mapping int to float
        Closeness centrality score for each node
    """
    ...

def pagerank_py(
    graph: Any,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict[int, float]:
    """
    Compute PageRank centrality for all nodes.

    Parameters
    ----------
    graph : Graph object
        Directed graph
    damping : float, optional
        Damping factor (default: 0.85)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    tol : float, optional
        Convergence tolerance (default: 1e-6)

    Returns
    -------
    dict mapping int to float
        PageRank score for each node
    """
    ...

def louvain_communities_py(graph: Any) -> List[List[int]]:
    """
    Detect communities using the Louvain algorithm.

    Parameters
    ----------
    graph : Graph object
        Undirected graph

    Returns
    -------
    list of list of int
        Each inner list is a community (set of node indices)
    """
    ...

def label_propagation_py(graph: Any) -> List[List[int]]:
    """
    Detect communities using label propagation algorithm.

    Parameters
    ----------
    graph : Graph object
        Undirected graph

    Returns
    -------
    list of list of int
        Each inner list is a community
    """
    ...

def modularity_py(
    graph: Any,
    communities: List[List[int]],
) -> float:
    """
    Compute the modularity of a community partition.

    Modularity measures the quality of a community partition; higher values
    indicate stronger community structure.

    Parameters
    ----------
    graph : Graph object
        Undirected graph
    communities : list of list of int
        Community partition

    Returns
    -------
    float
        Modularity score in [-1, 1]
    """
    ...

def diameter_py(graph: Any) -> Optional[float]:
    """
    Compute the diameter of a graph (longest shortest path).

    Parameters
    ----------
    graph : Graph object
        Connected graph

    Returns
    -------
    float or None
        Diameter of the graph; None if the graph is not connected
    """
    ...

def clustering_coefficient_py(graph: Any) -> Dict[str, Any]:
    """
    Compute the clustering coefficient for all nodes and the global average.

    The clustering coefficient of a node measures how close its neighbors
    are to being a complete graph.

    Parameters
    ----------
    graph : Graph object
        Undirected graph

    Returns
    -------
    dict
        Dictionary with 'global' (average) and 'local' (per-node dict) coefficients
    """
    ...

def graph_density_py(graph: Any) -> float:
    """
    Compute the density of a graph.

    Density = actual edges / maximum possible edges.

    Parameters
    ----------
    graph : Graph object

    Returns
    -------
    float
        Density in [0, 1]
    """
    ...

def minimum_spanning_tree_py(graph: Any) -> Any:
    """
    Compute the minimum spanning tree (Kruskal's algorithm).

    Parameters
    ----------
    graph : Graph object
        Weighted undirected graph

    Returns
    -------
    Graph object
        Minimum spanning tree
    """
    ...

# =============================================================================
# Machine Learning Metrics Module
# =============================================================================

def accuracy_score_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : ndarray of int
        True class labels
    y_pred : ndarray of int
        Predicted class labels

    Returns
    -------
    float
        Fraction of correctly classified samples in [0, 1]
    """
    ...

def precision_score_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    average: str = "macro",
) -> float:
    """
    Compute precision score.

    Parameters
    ----------
    y_true : ndarray of int
        True class labels
    y_pred : ndarray of int
        Predicted class labels
    average : str, optional
        Averaging strategy: 'macro', 'micro', 'weighted' (default: 'macro')

    Returns
    -------
    float
        Precision score
    """
    ...

def recall_score_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    average: str = "macro",
) -> float:
    """
    Compute recall (sensitivity) score.

    Parameters
    ----------
    y_true : ndarray of int
        True class labels
    y_pred : ndarray of int
        Predicted class labels
    average : str, optional
        Averaging strategy: 'macro', 'micro', 'weighted' (default: 'macro')

    Returns
    -------
    float
        Recall score
    """
    ...

def f1_score_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    average: str = "macro",
) -> float:
    """
    Compute F1 score (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true : ndarray of int
        True class labels
    y_pred : ndarray of int
        Predicted class labels
    average : str, optional
        Averaging strategy: 'macro', 'micro', 'weighted' (default: 'macro')

    Returns
    -------
    float
        F1 score
    """
    ...

def fbeta_score_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    beta: float,
    average: str = "macro",
) -> float:
    """
    Compute F-beta score (generalized F-measure).

    Parameters
    ----------
    y_true : ndarray of int
        True class labels
    y_pred : ndarray of int
        Predicted class labels
    beta : float
        Weight of recall relative to precision; beta=1 gives F1
    average : str, optional
        Averaging strategy: 'macro', 'micro', 'weighted' (default: 'macro')

    Returns
    -------
    float
        F-beta score
    """
    ...

def confusion_matrix_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Compute the confusion matrix.

    Parameters
    ----------
    y_true : ndarray of int
        True class labels
    y_pred : ndarray of int
        Predicted class labels

    Returns
    -------
    ndarray of shape (n_classes, n_classes)
        Confusion matrix where entry [i, j] is the number of samples
        with true class i predicted as class j
    """
    ...

def roc_curve_py(
    y_true: NDArray[np.int64],
    y_score: NDArray[np.float64],
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true : ndarray of int
        True binary labels (0 or 1)
    y_score : ndarray of float
        Predicted probability scores for the positive class

    Returns
    -------
    dict
        Dictionary with 'fpr', 'tpr', and 'thresholds' arrays
    """
    ...

def roc_auc_score_py(
    y_true: NDArray[np.int64],
    y_score: NDArray[np.float64],
) -> float:
    """
    Compute Area Under the ROC Curve (AUC-ROC).

    Parameters
    ----------
    y_true : ndarray of int
        True binary labels (0 or 1)
    y_score : ndarray of float
        Predicted probability scores for the positive class

    Returns
    -------
    float
        AUC-ROC score in [0, 1]
    """
    ...

def log_loss_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Compute log loss (cross-entropy loss) for classification.

    Parameters
    ----------
    y_true : ndarray of int
        True binary labels (0 or 1)
    y_pred : ndarray of float
        Predicted probabilities for the positive class

    Returns
    -------
    float
        Log loss (lower is better)
    """
    ...

def matthews_corrcoef_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """
    Compute Matthews Correlation Coefficient (MCC).

    Parameters
    ----------
    y_true : ndarray of int
        True binary labels
    y_pred : ndarray of int
        Predicted binary labels

    Returns
    -------
    float
        MCC in [-1, 1]
    """
    ...

def balanced_accuracy_score_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """
    Compute balanced accuracy (average recall per class).

    Parameters
    ----------
    y_true : ndarray of int
        True class labels
    y_pred : ndarray of int
        Predicted class labels

    Returns
    -------
    float
        Balanced accuracy in [0, 1]
    """
    ...

def cohen_kappa_score_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """
    Compute Cohen's kappa score (inter-rater agreement).

    Parameters
    ----------
    y_true : ndarray of int
        First set of ratings
    y_pred : ndarray of int
        Second set of ratings

    Returns
    -------
    float
        Kappa score
    """
    ...

def mean_squared_error_py(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Compute Mean Squared Error (MSE) for regression.

    Parameters
    ----------
    y_true : ndarray of float
        True values
    y_pred : ndarray of float
        Predicted values

    Returns
    -------
    float
        MSE (lower is better)
    """
    ...

def mean_absolute_error_py(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Compute Mean Absolute Error (MAE) for regression.

    Parameters
    ----------
    y_true : ndarray of float
        True values
    y_pred : ndarray of float
        Predicted values

    Returns
    -------
    float
        MAE (lower is better)
    """
    ...

def r2_score_py(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Compute R^2 (coefficient of determination) for regression.

    Parameters
    ----------
    y_true : ndarray of float
        True values
    y_pred : ndarray of float
        Predicted values

    Returns
    -------
    float
        R^2 score; 1.0 = perfect fit, 0.0 = constant baseline
    """
    ...

def mape_py(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : ndarray of float
        True values (must not contain zeros)
    y_pred : ndarray of float
        Predicted values

    Returns
    -------
    float
        MAPE as a percentage (lower is better)
    """
    ...

def explained_variance_score_py(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Compute explained variance regression score.

    Parameters
    ----------
    y_true : ndarray of float
        True values
    y_pred : ndarray of float
        Predicted values

    Returns
    -------
    float
        Explained variance score in [0, 1]
    """
    ...

def adjusted_rand_index_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """
    Compute Adjusted Rand Index (ARI) for clustering evaluation.

    Parameters
    ----------
    y_true : ndarray of int
        True cluster labels
    y_pred : ndarray of int
        Predicted cluster labels

    Returns
    -------
    float
        ARI in [-1, 1]
    """
    ...

def nmi_score_py(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """
    Compute Normalized Mutual Information (NMI) for clustering evaluation.

    Parameters
    ----------
    y_true : ndarray of int
        True cluster labels
    y_pred : ndarray of int
        Predicted cluster labels

    Returns
    -------
    float
        NMI in [0, 1]
    """
    ...

def ndcg_score_py(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
    k: Optional[int] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) for ranking evaluation.

    Parameters
    ----------
    y_true : ndarray of float
        True relevance scores
    y_score : ndarray of float
        Predicted scores (higher = more relevant)
    k : int, optional
        Only consider top-k elements. If None, use all elements.

    Returns
    -------
    float
        NDCG score in [0, 1]
    """
    ...

def mrr_py(
    y_true: NDArray[np.int64],
    y_score: NDArray[np.float64],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for ranking/retrieval evaluation.

    Parameters
    ----------
    y_true : ndarray of int
        Binary relevance labels (1 = relevant, 0 = not relevant)
    y_score : ndarray of float
        Predicted scores

    Returns
    -------
    float
        MRR score in (0, 1]
    """
    ...

# =============================================================================
# File I/O Module
# =============================================================================

def read_csv_py(
    path: str,
    delimiter: str = ",",
    has_header: bool = True,
) -> NDArray[np.float64]:
    """
    Read a CSV file into a numpy array.

    Parameters
    ----------
    path : str
        Path to the CSV file
    delimiter : str, optional
        Field delimiter (default: ',')
    has_header : bool, optional
        If True, skip the first line as header (default: True)

    Returns
    -------
    ndarray of float
        Data matrix read from the CSV file
    """
    ...

def write_csv_py(
    path: str,
    data: NDArray[np.float64],
    delimiter: str = ",",
    header: Optional[str] = None,
) -> None:
    """
    Write a numpy array to a CSV file.

    Parameters
    ----------
    path : str
        Output file path
    data : ndarray
        2-D array to write
    delimiter : str, optional
        Field delimiter (default: ',')
    header : str, optional
        Header line to write as the first row
    """
    ...

def read_matrix_market_sparse_py(path: str) -> Dict[str, Any]:
    """
    Read a sparse matrix in Matrix Market (.mtx) format.

    Parameters
    ----------
    path : str
        Path to the .mtx file

    Returns
    -------
    dict
        Sparse matrix in COO dict representation
    """
    ...

def write_matrix_market_sparse_py(
    path: str,
    sparse_dict: Dict[str, Any],
) -> None:
    """
    Write a sparse matrix to Matrix Market (.mtx) format.

    Parameters
    ----------
    path : str
        Output file path
    sparse_dict : dict
        Sparse matrix dict
    """
    ...

def read_matrix_market_dense_py(path: str) -> NDArray[np.float64]:
    """
    Read a dense matrix from a Matrix Market file.

    Parameters
    ----------
    path : str
        Path to the .mtx file (dense format)

    Returns
    -------
    ndarray of float64
        Dense matrix
    """
    ...

def write_matrix_market_dense_py(
    path: str,
    data: NDArray[np.float64],
) -> None:
    """
    Write a dense matrix to Matrix Market format.

    Parameters
    ----------
    path : str
        Output file path
    data : ndarray
        2-D array to write
    """
    ...

def save_array_py(
    path: str,
    data: NDArray[np.float64],
    format: str = "npy",
) -> None:
    """
    Save a numpy array to a file.

    Parameters
    ----------
    path : str
        Output file path
    data : ndarray
        Array to save
    format : str, optional
        File format: 'npy', 'csv', or 'txt' (default: 'npy')
    """
    ...

def load_array_py(
    path: str,
    format: str = "npy",
) -> NDArray[np.float64]:
    """
    Load a numpy array from a file.

    Parameters
    ----------
    path : str
        Input file path
    format : str, optional
        File format: 'npy', 'csv', or 'txt' (default: 'npy')

    Returns
    -------
    ndarray of float64
        Loaded array
    """
    ...

def read_wav_py(path: str) -> Dict[str, Any]:
    """
    Read a WAV audio file.

    Parameters
    ----------
    path : str
        Path to the .wav file

    Returns
    -------
    dict
        Dictionary with 'samplerate' (int) and 'data' (ndarray) keys
    """
    ...

def write_wav_py(
    path: str,
    samplerate: int,
    data: NDArray[np.float64],
) -> None:
    """
    Write audio data to a WAV file.

    Parameters
    ----------
    path : str
        Output file path
    samplerate : int
        Sample rate in Hz
    data : ndarray of float64
        Audio samples (1-D array)
    """
    ...

# =============================================================================
# Datasets Module
# =============================================================================

def load_iris_py() -> Dict[str, Any]:
    """
    Load the Iris dataset.

    Returns
    -------
    dict
        Dictionary with 'data', 'target', 'feature_names', 'target_names'
    """
    ...

def load_boston_py() -> Dict[str, Any]:
    """
    Load the Boston Housing dataset.

    Returns
    -------
    dict
        Dictionary with 'data', 'target', and 'feature_names' keys
    """
    ...

def load_diabetes_py() -> Dict[str, Any]:
    """
    Load the Diabetes dataset.

    Returns
    -------
    dict
        Dictionary with 'data', 'target', and 'feature_names' keys
    """
    ...

def load_breast_cancer_py() -> Dict[str, Any]:
    """
    Load the Breast Cancer Wisconsin dataset.

    Returns
    -------
    dict
        Dictionary with 'data', 'target', 'feature_names', and 'target_names'
    """
    ...

def load_digits_py() -> Dict[str, Any]:
    """
    Load the Digits handwritten digit dataset.

    Returns
    -------
    dict
        Dictionary with 'data', 'target', and 'images' keys
    """
    ...

def make_classification_py(
    n_samples: int = 100,
    n_features: int = 20,
    n_informative: int = 2,
    n_classes: int = 2,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Generate a random n-class classification dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    n_features : int, optional
        Total number of features (default: 20)
    n_informative : int, optional
        Number of informative features (default: 2)
    n_classes : int, optional
        Number of classes (default: 2)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with 'X' (feature matrix) and 'y' (class labels)
    """
    ...

def make_regression_py(
    n_samples: int = 100,
    n_features: int = 100,
    n_informative: int = 10,
    noise: float = 0.0,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Generate a random regression dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    n_features : int, optional
        Number of features (default: 100)
    n_informative : int, optional
        Number of informative features (default: 10)
    noise : float, optional
        Gaussian noise standard deviation (default: 0.0)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with 'X' (feature matrix) and 'y' (target values)
    """
    ...

def make_blobs_py(
    n_samples: int = 100,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Generate isotropic Gaussian blobs for clustering.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    n_features : int, optional
        Number of features (default: 2)
    centers : int, optional
        Number of cluster centers (default: 3)
    cluster_std : float, optional
        Standard deviation of each cluster (default: 1.0)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with 'X' (feature matrix) and 'y' (cluster labels)
    """
    ...

def make_moons_py(
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Generate two interleaving half circles (moon shapes).

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    noise : float, optional
        Gaussian noise standard deviation (default: 0.1)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with 'X' (n x 2 feature matrix) and 'y' (binary labels)
    """
    ...

def make_circles_py(
    n_samples: int = 100,
    noise: float = 0.1,
    factor: float = 0.8,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Generate a large circle containing a smaller circle.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    noise : float, optional
        Gaussian noise standard deviation (default: 0.1)
    factor : float, optional
        Scale factor between inner and outer circle (default: 0.8)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with 'X' (n x 2 feature matrix) and 'y' (binary labels)
    """
    ...

def make_spirals_py(
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Generate spiral-shaped clusters.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    noise : float, optional
        Gaussian noise standard deviation (default: 0.1)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with 'X' (n x 2 feature matrix) and 'y' (binary labels)
    """
    ...

def make_swiss_roll_py(
    n_samples: int = 100,
    noise: float = 0.0,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Generate a Swiss roll dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    noise : float, optional
        Gaussian noise standard deviation (default: 0.0)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with 'X' (n x 3 feature matrix) and 't' (1D manifold position)
    """
    ...

def make_s_curve_py(
    n_samples: int = 100,
    noise: float = 0.0,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Generate an S-curve dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (default: 100)
    noise : float, optional
        Gaussian noise standard deviation (default: 0.0)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with 'X' (n x 3 feature matrix) and 't' (1D manifold position)
    """
    ...

def train_test_split_py(
    X: NDArray,
    y: NDArray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Dict[str, NDArray]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : ndarray
        Feature matrix of shape (n_samples, n_features)
    y : ndarray
        Target array of shape (n_samples,)
    test_size : float, optional
        Proportion of dataset to include in the test split (default: 0.2)
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Dictionary with 'X_train', 'X_test', 'y_train', 'y_test' arrays
    """
    ...

def k_fold_split_py(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = False,
    random_state: Optional[int] = None,
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate k-fold cross-validation indices.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_splits : int, optional
        Number of folds (default: 5)
    shuffle : bool, optional
        Whether to shuffle before splitting (default: False)
    random_state : int, optional
        Random seed

    Returns
    -------
    list of (list of int, list of int)
        Each element is a (train_indices, test_indices) tuple
    """
    ...

def min_max_scale_py(
    data: NDArray[np.float64],
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> NDArray[np.float64]:
    """
    Scale features to a given range using min-max normalization.

    Parameters
    ----------
    data : ndarray
        Input data to scale
    feature_range : tuple of (float, float), optional
        Target range (default: (0.0, 1.0))

    Returns
    -------
    ndarray
        Scaled data
    """
    ...

# =============================================================================
# Data Transformation Module
# =============================================================================

def normalize_array_py(
    data: NDArray[np.float64],
    norm: str = "l2",
) -> NDArray[np.float64]:
    """
    Normalize an array (rows) using the specified norm.

    Parameters
    ----------
    data : ndarray of shape (n, m)
        Input data matrix
    norm : str, optional
        Normalization type: 'l1', 'l2', or 'max' (default: 'l2')

    Returns
    -------
    ndarray
        Normalized data where each row has unit norm
    """
    ...

def normalize_vector_py(
    v: NDArray[np.float64],
    norm: str = "l2",
) -> NDArray[np.float64]:
    """
    Normalize a 1-D vector using the specified norm.

    Parameters
    ----------
    v : ndarray of shape (n,)
        Input vector
    norm : str, optional
        Normalization type: 'l1', 'l2', or 'max' (default: 'l2')

    Returns
    -------
    ndarray
        Normalized vector
    """
    ...

def binarize_py(
    data: NDArray[np.float64],
    threshold: float = 0.0,
) -> NDArray[np.float64]:
    """
    Binarize data according to a threshold.

    Parameters
    ----------
    data : ndarray
        Input data
    threshold : float, optional
        Values above this become 1.0, at or below become 0.0 (default: 0.0)

    Returns
    -------
    ndarray
        Binary array with the same shape as data
    """
    ...

def discretize_equal_width_py(
    data: NDArray[np.float64],
    n_bins: int,
) -> NDArray[np.int64]:
    """
    Discretize continuous data into equal-width bins.

    Parameters
    ----------
    data : ndarray
        Input 1-D continuous data
    n_bins : int
        Number of bins

    Returns
    -------
    ndarray of int
        Bin indices for each data point (0-based)
    """
    ...

def discretize_equal_frequency_py(
    data: NDArray[np.float64],
    n_bins: int,
) -> NDArray[np.int64]:
    """
    Discretize continuous data into equal-frequency (quantile) bins.

    Parameters
    ----------
    data : ndarray
        Input 1-D continuous data
    n_bins : int
        Number of bins (approximately equal number of samples per bin)

    Returns
    -------
    ndarray of int
        Bin indices for each data point (0-based)
    """
    ...

def log_transform_py(
    data: NDArray[np.float64],
    shift: float = 0.0,
) -> NDArray[np.float64]:
    """
    Apply log transformation: log(data + shift).

    Parameters
    ----------
    data : ndarray
        Input data (must be positive after adding shift)
    shift : float, optional
        Value to add before taking log (default: 0.0)

    Returns
    -------
    ndarray
        Log-transformed data
    """
    ...

def power_transform_py(
    data: NDArray[np.float64],
    power: float,
) -> NDArray[np.float64]:
    """
    Apply power transformation: data^power.

    Parameters
    ----------
    data : ndarray
        Input data
    power : float
        Exponent to apply

    Returns
    -------
    ndarray
        Power-transformed data
    """
    ...

# =============================================================================
# Text Processing Module
# =============================================================================

def levenshtein_distance_py(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns
    -------
    int
        Edit distance (0 = identical strings)
    """
    ...

def cosine_similarity_py(v1: List[float], v2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    v1 : list of float
        First vector
    v2 : list of float
        Second vector (must have same length as v1)

    Returns
    -------
    float
        Cosine similarity in [-1, 1]
    """
    ...

def jaccard_similarity_py(s1: str, s2: str) -> float:
    """
    Compute Jaccard similarity between two strings based on character sets.

    Parameters
    ----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns
    -------
    float
        Jaccard similarity in [0, 1]
    """
    ...

def strip_html_tags_py(text: str) -> str:
    """
    Remove HTML tags from a string.

    Parameters
    ----------
    text : str
        Input string potentially containing HTML tags

    Returns
    -------
    str
        String with all HTML tags removed
    """
    ...

def replace_urls_py(text: str, replacement: str = "<URL>") -> str:
    """
    Replace URL patterns in text with a placeholder string.

    Parameters
    ----------
    text : str
        Input text
    replacement : str, optional
        String to replace URLs with (default: '<URL>')

    Returns
    -------
    str
        Text with URLs replaced
    """
    ...

def replace_emails_py(text: str, replacement: str = "<EMAIL>") -> str:
    """
    Replace email address patterns in text with a placeholder string.

    Parameters
    ----------
    text : str
        Input text
    replacement : str, optional
        Replacement string (default: '<EMAIL>')

    Returns
    -------
    str
        Text with email addresses replaced
    """
    ...

def expand_contractions_py(text: str) -> str:
    """
    Expand common English contractions in text.

    Parameters
    ----------
    text : str
        Input text with contractions (e.g., "don't", "can't", "it's")

    Returns
    -------
    str
        Text with contractions expanded
    """
    ...

def normalize_unicode_py(text: str) -> str:
    """
    Normalize Unicode characters in text to NFC form.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
        Unicode-normalized text
    """
    ...

def normalize_whitespace_py(text: str) -> str:
    """
    Normalize whitespace in text: collapse multiple spaces to one.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
        Text with normalized whitespace
    """
    ...

def remove_accents_py(text: str) -> str:
    """
    Remove accent characters (diacritics) from text.

    Parameters
    ----------
    text : str
        Input text with accented characters

    Returns
    -------
    str
        Text with accents removed (e.g., 'e with accent' becomes 'e')
    """
    ...

# =============================================================================
# Computer Vision Module
# =============================================================================

def rgb_to_grayscale_py(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert an RGB image to grayscale.

    Parameters
    ----------
    image : ndarray of uint8 and shape (H, W, 3)
        Input RGB image

    Returns
    -------
    ndarray of uint8 and shape (H, W)
        Grayscale image
    """
    ...

def rgb_to_hsv_py(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """
    Convert an RGB image to HSV (Hue, Saturation, Value) color space.

    Parameters
    ----------
    image : ndarray of uint8 and shape (H, W, 3)
        Input RGB image

    Returns
    -------
    ndarray of float64 and shape (H, W, 3)
        HSV image with H in [0, 360], S and V in [0, 1]
    """
    ...

def sobel_edges_py(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """
    Detect edges using the Sobel operator.

    Parameters
    ----------
    image : ndarray of uint8
        Input grayscale image

    Returns
    -------
    ndarray of float64
        Edge magnitude image
    """
    ...

def canny_edges_py(
    image: NDArray[np.uint8],
    low_threshold: float,
    high_threshold: float,
) -> NDArray[np.uint8]:
    """
    Detect edges using the Canny edge detector.

    Parameters
    ----------
    image : ndarray of uint8
        Input grayscale image
    low_threshold : float
        Lower hysteresis threshold
    high_threshold : float
        Upper hysteresis threshold

    Returns
    -------
    ndarray of uint8
        Binary edge image (255 = edge, 0 = background)
    """
    ...

def prewitt_edges_py(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """
    Detect edges using the Prewitt operator.

    Parameters
    ----------
    image : ndarray of uint8
        Input grayscale image

    Returns
    -------
    ndarray of float64
        Edge magnitude image
    """
    ...

def laplacian_edges_py(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """
    Detect edges using the Laplacian operator.

    Parameters
    ----------
    image : ndarray of uint8
        Input grayscale image

    Returns
    -------
    ndarray of float64
        Edge response image
    """
    ...

def equalize_histogram_py(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Apply global histogram equalization to a grayscale image.

    Parameters
    ----------
    image : ndarray of uint8
        Input grayscale image of shape (H, W)

    Returns
    -------
    ndarray of uint8
        Histogram-equalized image
    """
    ...

def clahe_py(
    image: NDArray[np.uint8],
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
) -> NDArray[np.uint8]:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters
    ----------
    image : ndarray of uint8
        Input grayscale image
    clip_limit : float
        Threshold for contrast limiting
    tile_grid_size : tuple of (int, int)
        Size of grid for histogram equalization tiles

    Returns
    -------
    ndarray of uint8
        Contrast-enhanced image
    """
    ...

def normalize_brightness_py(
    image: NDArray[np.uint8],
    target_mean: float,
) -> NDArray[np.uint8]:
    """
    Normalize the brightness of an image to a target mean value.

    Parameters
    ----------
    image : ndarray of uint8
        Input image
    target_mean : float
        Target mean pixel intensity

    Returns
    -------
    ndarray of uint8
        Brightness-normalized image
    """
    ...

def unsharp_mask_py(
    image: NDArray[np.uint8],
    sigma: float,
    strength: float,
) -> NDArray[np.uint8]:
    """
    Sharpen an image using the unsharp masking technique.

    Parameters
    ----------
    image : ndarray of uint8
        Input image
    sigma : float
        Gaussian blur sigma for creating the mask
    strength : float
        Sharpening strength factor

    Returns
    -------
    ndarray of uint8
        Sharpened image
    """
    ...

def gaussian_blur_py(
    image: NDArray[np.uint8],
    sigma: float,
    kernel_size: Optional[int] = None,
) -> NDArray[np.uint8]:
    """
    Apply Gaussian blur to a vision image.

    Parameters
    ----------
    image : ndarray of uint8
        Input image
    sigma : float
        Standard deviation of the Gaussian kernel
    kernel_size : int, optional
        Size of the kernel; if None, derived from sigma

    Returns
    -------
    ndarray of uint8
        Blurred image
    """
    ...

def detect_and_compute_sift_py(
    image: NDArray[np.uint8],
) -> Dict[str, Any]:
    """
    Detect SIFT keypoints and compute descriptors.

    Parameters
    ----------
    image : ndarray of uint8
        Input grayscale image

    Returns
    -------
    dict
        Dictionary with 'keypoints' (list of (x, y, size, angle)) and
        'descriptors' (ndarray of shape (n_kp, 128))
    """
    ...

def labels_to_color_image_py(
    labels: NDArray[np.int32],
) -> NDArray[np.uint8]:
    """
    Convert a labeled segmentation map to a color image.

    Parameters
    ----------
    labels : ndarray of int32
        Labeled image where each unique value is a segment

    Returns
    -------
    ndarray of uint8 and shape (H, W, 3)
        RGB color image with each label assigned a distinct color
    """
    ...

def find_homography_py(
    src_points: NDArray[np.float64],
    dst_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Find a homography matrix mapping source points to destination points.

    Parameters
    ----------
    src_points : ndarray of shape (n, 2)
        Source image points (at least 4 non-collinear points required)
    dst_points : ndarray of shape (n, 2)
        Destination image points

    Returns
    -------
    ndarray of shape (3, 3)
        Homography matrix H
    """
    ...

# =============================================================================
# Async Operations Module
# =============================================================================

def fft_async(data: NDArray[np.float64]) -> Any:
    """
    Asynchronous FFT operation for large arrays (must be awaited).

    Parameters
    ----------
    data : ndarray of float64
        Real-valued input data

    Returns
    -------
    coroutine
        Awaitable returning a complex-valued result dict
    """
    ...

def svd_async(
    matrix: NDArray[np.float64],
    full_matrices: Optional[bool] = None,
) -> Any:
    """
    Asynchronous SVD decomposition for large matrices (must be awaited).

    Parameters
    ----------
    matrix : ndarray of shape (m, n)
        Input matrix
    full_matrices : bool, optional
        If True, compute full U and Vt matrices (default: True)

    Returns
    -------
    coroutine
        Awaitable returning a dict with 'U', 'S', 'Vt' arrays
    """
    ...

def qr_async(matrix: NDArray[np.float64]) -> Any:
    """
    Asynchronous QR decomposition for large matrices (must be awaited).

    Parameters
    ----------
    matrix : ndarray of shape (m, n)
        Input matrix

    Returns
    -------
    coroutine
        Awaitable returning a dict with 'Q' and 'R' arrays
    """
    ...

def quad_async(
    func: Any,
    a: float,
    b: float,
    epsabs: Optional[float] = None,
    epsrel: Optional[float] = None,
) -> Any:
    """
    Asynchronous numerical integration (must be awaited).

    Parameters
    ----------
    func : callable
        Python callable f(x) -> float
    a : float
        Lower integration limit
    b : float
        Upper integration limit
    epsabs : float, optional
        Absolute error tolerance (default: 1e-8)
    epsrel : float, optional
        Relative error tolerance (default: 1e-8)

    Returns
    -------
    coroutine
        Awaitable returning a dict with 'value' and 'error' keys
    """
    ...

def minimize_async(
    func: Any,
    x0: NDArray[np.float64],
    method: Optional[str] = None,
    maxiter: Optional[int] = None,
) -> Any:
    """
    Asynchronous optimization for expensive objective functions (must be awaited).

    Parameters
    ----------
    func : callable
        Objective function f(x) -> float
    x0 : ndarray
        Initial guess
    method : str, optional
        Optimization method (e.g., 'nelder-mead', 'bfgs')
    maxiter : int, optional
        Maximum number of iterations

    Returns
    -------
    coroutine
        Awaitable returning an optimization result dict
    """
    ...

# =============================================================================
# Batch Statistics Module
# =============================================================================

def stats_summary(data: List[float]) -> Tuple[float, float, float]:
    """
    Compute a quick summary of basic statistics.

    Parameters
    ----------
    data : list of float
        Input data values

    Returns
    -------
    tuple of (float, float, float)
        Tuple of (mean, variance, std)
    """
    ...

def batch_descriptive_stats(
    arrays: List[List[float]],
) -> List[Dict[str, float]]:
    """
    Compute descriptive statistics for multiple arrays in parallel.

    Parameters
    ----------
    arrays : list of list of float
        List of data arrays

    Returns
    -------
    list of dict
        One dict per input array with keys: 'mean', 'variance', 'std',
        'min', 'max', 'median', 'count'
    """
    ...

def batch_correlation(
    arrays: List[List[float]],
) -> List[List[float]]:
    """
    Compute pairwise Pearson correlation matrix for multiple arrays.

    Parameters
    ----------
    arrays : list of list of float
        List of data arrays (must all have the same length)

    Returns
    -------
    list of list of float
        Correlation matrix as nested list
    """
    ...

def batch_pdf_eval(
    distribution: str,
    params: List[float],
    x_values: List[float],
) -> List[float]:
    """
    Evaluate a probability density function at multiple points.

    Parameters
    ----------
    distribution : str
        Name of the distribution (e.g., 'norm', 'gamma', 'beta')
    params : list of float
        Distribution parameters (vary by distribution)
    x_values : list of float
        Points at which to evaluate the PDF

    Returns
    -------
    list of float
        PDF values at each point
    """
    ...

# =============================================================================
# Batch Linear Algebra Module
# =============================================================================

def batch_matmul_py(
    a_list: List[List[List[float]]],
    b_list: List[List[List[float]]],
) -> List[List[List[float]]]:
    """
    Compute batch matrix multiplication: result[i] = a_list[i] @ b_list[i].

    Processes pairs of matrices in parallel using Rayon.

    Parameters
    ----------
    a_list : list of 2-D matrix (as list of list of float)
        List of left-hand matrices
    b_list : list of 2-D matrix (as list of list of float)
        List of right-hand matrices (must have same length as a_list)

    Returns
    -------
    list of 2-D matrix (as list of list of float)
        List of result matrices
    """
    ...

def batch_svd_py(
    matrices: List[List[List[float]]],
) -> List[Dict[str, List[List[float]]]]:
    """
    Compute SVD for a batch of matrices in parallel.

    Parameters
    ----------
    matrices : list of 2-D matrix (as list of list of float)
        List of matrices to decompose

    Returns
    -------
    list of dict
        Each dict has 'u', 's', 'vt' keys with the SVD components
    """
    ...

def batch_solve_py(
    a_list: List[List[List[float]]],
    b_list: List[List[float]],
) -> List[List[float]]:
    """
    Solve a batch of linear systems Ax = b in parallel.

    Parameters
    ----------
    a_list : list of 2-D matrix (as list of list of float)
        List of coefficient matrices (each must be square)
    b_list : list of list of float
        List of right-hand side vectors

    Returns
    -------
    list of list of float
        List of solution vectors
    """
    ...

def batch_matrix_norm_py(
    matrices: List[List[List[float]]],
    ord: str = "fro",
) -> List[float]:
    """
    Compute matrix norms for a batch of matrices in parallel.

    Parameters
    ----------
    matrices : list of 2-D matrix (as list of list of float)
        List of input matrices
    ord : str, optional
        Norm type: 'fro' (Frobenius), '1', 'inf', '2' (default: 'fro')

    Returns
    -------
    list of float
        Norm value for each input matrix
    """
    ...

# =============================================================================
# Polynomial Fitting (Statistics Extension)
# =============================================================================

def polyfit_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    degree: int,
) -> Dict[str, Any]:
    """
    Fit a polynomial of the specified degree to data points.

    Uses least-squares to find coefficients of a polynomial p(x) = c_0 + c_1*x
    + ... + c_d*x^d that minimizes the sum of squared residuals.

    Parameters
    ----------
    x : ndarray of float64
        Independent variable (x-coordinates)
    y : ndarray of float64
        Dependent variable (y-coordinates; must have same length as x)
    degree : int
        Degree of the polynomial to fit (must be >= 1)

    Returns
    -------
    dict
        Dictionary with:
        - 'coefficients': ndarray of shape (degree+1,) in ascending order
        - 'residuals': float, sum of squared residuals
        - 'rank': int, rank of the Vandermonde matrix

    Examples
    --------
    >>> x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([1.0, 3.0, 7.0, 13.0, 21.0])  # y = x^2 + x + 1
    >>> result = scirs2.polyfit_py(x, y, degree=2)
    """
    ...
