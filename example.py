from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from simple_blocks import block


def simple_matmul(A: NDArray, B: NDArray) -> NDArray:
    """
    Matrix multiplication using the @ operator.
    This function is decorated to use block-based tiling.

    Args:
        A: First matrix of shape (m, k) as a numpy array
        B: Second matrix of shape (k, n) as a numpy array

    Returns:
        C: Result matrix of shape (m, n) as a numpy array
    """
    return A @ B


def benchmark_matmul(
    A: NDArray, B: NDArray, block_sizes: Optional[Tuple[Tuple[int, int], ...]] = None
) -> dict:
    """
    Benchmark matrix multiplication with different block sizes.

    Args:
        A: First matrix of shape (m, k)
        B: Second matrix of shape (k, n)
        block_sizes: List of block size tuples to test, defaults to some common sizes

    Returns:
        Dictionary mapping block sizes to execution times
    """
    import time

    if block_sizes is None:
        block_sizes = ((16, 16), (32, 32), (64, 64), (128, 128))

    results: dict = {}

    # Get reference result for verification
    reference: NDArray = A @ B

    for bm, bn in block_sizes:
        # Create a new decorated function with this block size
        @block(bm, bn)
        def test_matmul(A: NDArray, B: NDArray) -> NDArray:
            return A @ B

        # Warm-up run
        _ = test_matmul(A, B)

        # Timed run
        start_time: float = time.time()
        result: NDArray = test_matmul(A, B)
        end_time: float = time.time()

        # Verify correctness
        is_correct: bool = np.allclose(result, reference)

        results[(bm, bn)] = {"time": end_time - start_time, "correct": is_correct}

    return results


# Usage example
if __name__ == "__main__":
    # Create example matrices
    A: NDArray = np.random.rand(1024, 512)
    B: NDArray = np.random.rand(512, 768)

    # Perform block-based matrix multiplication
    C: NDArray = simple_matmul(A, B)

    # Verify result against numpy's implementation
    expected: NDArray = A @ B
    print(f"Result is correct: {np.allclose(C, expected)}")

    # Optional benchmark
    results = benchmark_matmul(A, B)
    for (bm, bn), data in results.items():
        print(
            f"Block size ({bm, bn}): {data['time']:.4f}s - Correct: {data['correct']}"
        )
