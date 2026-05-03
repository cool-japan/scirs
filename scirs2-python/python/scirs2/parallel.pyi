"""Type stubs for scirs2 parallel batch-processing functions."""

from typing import List

def parallel_map_mean(arrays: List[List[float]]) -> List[float]:
    """Compute the arithmetic mean of each sub-array in parallel.

    The GIL is released during Rayon computation, allowing Python threads to
    run concurrently.  Empty sub-arrays return ``0.0``.

    Args:
        arrays: A list of float arrays.  One mean value is produced per element.

    Returns:
        A list of ``float`` mean values with the same length as *arrays*.
    """
    ...

def parallel_batch_matvec(
    matrices: List[List[float]],
    vectors: List[List[float]],
    n_rows: int,
    n_cols: int,
) -> List[List[float]]:
    """Batch matrix-vector multiplication in parallel.

    Each pair ``(matrices[i], vectors[i])`` computes ``A * v`` where ``A`` is
    stored in row-major order with shape ``(n_rows, n_cols)`` and ``v`` has
    length ``n_cols``.  The GIL is released during Rayon computation.

    Args:
        matrices: List of flat row-major matrices, each with
            ``n_rows * n_cols`` elements.
        vectors: List of vectors, each with ``n_cols`` elements.
        n_rows: Number of rows in each matrix.
        n_cols: Number of columns in each matrix (equals length of each vector).

    Returns:
        A list of result vectors, each of length ``n_rows``.

    Raises:
        ValueError: If *matrices* and *vectors* have different lengths, or if
            any matrix/vector has the wrong number of elements.
    """
    ...
