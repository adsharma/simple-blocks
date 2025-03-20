from functools import wraps
from typing import Callable, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

# Type variables for function signatures
T = TypeVar("T", bound=Callable[..., NDArray])
FloatArray = NDArray[np.float64]  # Type alias for float arrays


def block(block_m: int, block_n: int) -> Callable[[T], T]:
    """
    Decorator for implementing block-based tiling for matrix multiplication.

    Args:
        block_m: Block size for the rows (M dimension)
        block_n: Block size for the columns (N dimension)

    Returns:
        A decorator function that applies block-based tiling
    """

    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(A: NDArray, B: NDArray) -> NDArray:
            # Get dimensions
            m, k = A.shape
            k_check, n = B.shape

            # Ensure matrices are compatible for multiplication
            assert k == k_check, "Matrix dimensions incompatible for multiplication"

            # Initialize result matrix
            C: NDArray = np.zeros((m, n), dtype=A.dtype)

            # Process blocks
            for i in range(0, m, block_m):
                # Determine actual block size (handle edge cases)
                i_end: int = min(i + block_m, m)

                for j in range(0, n, block_n):
                    # Determine actual block size (handle edge cases)
                    j_end: int = min(j + block_n, n)

                    # Sub-blocks of C to be computed
                    C_block: NDArray = np.zeros((i_end - i, j_end - j), dtype=A.dtype)

                    # Process sub-blocks of A and B to compute C_block
                    for k_start in range(0, k, block_m):
                        k_end: int = min(k_start + block_m, k)

                        # Extract blocks from A and B
                        A_block: NDArray = A[i:i_end, k_start:k_end]
                        B_block: NDArray = B[k_start:k_end, j:j_end]

                        # Compute block multiplication using @ operator
                        C_block += A_block @ B_block

                    # Update the result matrix with the computed block
                    C[i:i_end, j:j_end] = C_block

            return C

        return cast(T, wrapper)  # Cast to maintain the type signature

    return decorator
