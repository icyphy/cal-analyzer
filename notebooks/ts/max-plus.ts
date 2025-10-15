/**
 * Max-Plus Algebra Library
 * 
 * A TypeScript library for max-plus algebra operations used in discrete event systems,
 * real-time systems, and CAL (Causality and Logical Time) theorem analysis.
 * 
 * Based on: https://github.com/msnhdyt/Max-Plus-Algebra
 * 
 * @author CAL Analyzer Project
 * @version 1.0.0
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * A matrix in max-plus algebra (2D array of numbers)
 */
export type Matrix = number[][];

/**
 * A vector in max-plus algebra (1D array of numbers)
 */
export type Vector = number[];

// ============================================================================
// CONSTANTS
// ============================================================================

/**
 * Epsilon (ε) - the empty element in max-plus algebra
 * Represents "nothing has happened yet" or "no constraint"
 */
export const EPS = Number.NEGATIVE_INFINITY;

/**
 * Infinity (∞) - represents "exhaustion of future events"
 */
export const INF = Number.POSITIVE_INFINITY;

// ============================================================================
// CORE MAX-PLUS OPERATIONS
// ============================================================================

/**
 * Max-plus addition (⊕): A ⊕ B = max(A, B) element-wise
 * 
 * @param A - First matrix
 * @param B - Second matrix (must have same dimensions as A)
 * @returns Matrix where each element is max(A[i][j], B[i][j])
 * @throws TypeError if matrices have different dimensions
 * 
 * @example
 * ```typescript
 * const A = [[1, 2], [3, 4]];
 * const B = [[5, 1], [2, 6]];
 * const result = oplus(A, B); // [[5, 2], [3, 6]]
 * ```
 */
export function oplus(A: Matrix, B: Matrix): Matrix {
  if (A.length === 0 || B.length === 0 || A[0].length === 0 || B[0].length === 0) {
    throw new TypeError('Matrices cannot be empty');
  }
  
  if (A.length !== B.length || A[0].length !== B[0].length) {
    throw new TypeError(`A and B must have the same shape: A.shape=[${A.length},${A[0].length}], B.shape=[${B.length},${B[0].length}]`);
  }
  
  const result: Matrix = [];
  for (let i = 0; i < A.length; i++) {
    result[i] = [];
    for (let j = 0; j < A[i].length; j++) {
      const aVal = A[i]?.[j] ?? EPS;
      const bVal = B[i]?.[j] ?? EPS;
      result[i][j] = Math.max(aVal, bVal);
    }
  }
  return result;
}

/**
 * Max-plus multiplication (⊗): A ⊗ B = max(A[i][k] + B[k][j]) for all k
 * 
 * @param A - First matrix (m × n)
 * @param B - Second matrix (n × p)
 * @returns Result matrix (m × p)
 * @throws TypeError if A's columns don't match B's rows
 * 
 * @example
 * ```typescript
 * const A = [[1, 2], [3, 4]];
 * const B = [[5, 1], [2, 6]];
 * const result = otimes(A, B); // [[7, 7], [9, 9]]
 * ```
 */
export function otimes(A: Matrix, B: Matrix): Matrix {
  if (A.length === 0 || B.length === 0 || A[0].length === 0 || B[0].length === 0) {
    throw new TypeError('Matrices cannot be empty');
  }
  
  if (A[0].length !== B.length) {
    throw new TypeError(`A's 2nd dimension does not match B's 1st dimension: A.shape=[${A.length},${A[0].length}], B.shape=[${B.length},${B[0].length}]`);
  }

  if (A.length === 1 && A[0].length === 1 || B.length === 1 && B[0].length === 1) {
    const aVal = A[0]?.[0] ?? EPS;
    const bVal = B[0]?.[0] ?? EPS;
    return [[aVal + bVal]];
  } else {
    const result: Matrix = [];
    const B_T = transpose(B);
    
    for (let i = 0; i < A.length; i++) {
      result[i] = [];
      const rowA = A[i];
      if (!rowA) continue;
      
      for (let j = 0; j < B_T.length; j++) {
        const colB = B_T[j];
        if (!colB) continue;
        
        result[i][j] = odot(rowA, colB);
      }
    }
    return result;
  }
}

/**
 * Max-plus dot product of two vectors: a ⊙ b = max(a[i] + b[i]) for all i
 * 
 * @param a - First vector
 * @param b - Second vector (must have same length as a)
 * @returns Scalar result
 * @throws Error if vectors have different lengths
 * 
 * @example
 * ```typescript
 * const a = [1, 2, 3];
 * const b = [4, 1, 2];
 * const result = odot(a, b); // 5 (max of [5, 3, 5])
 * ```
 */
export function odot(a: Vector, b: Vector): number {
  if (a.length !== b.length) {
    throw new Error(`Vector a and b must have the same length: a.length=${a.length}, b.length=${b.length}`);
  }
  
  const result: number[] = [];
  for (let i = 0; i < a.length; i++) {
    const aVal = a[i] ?? EPS;
    const bVal = b[i] ?? EPS;
    result[i] = aVal + bVal;
  }
  return Math.max(...result);
}

/**
 * Transpose a matrix (swap rows and columns)
 * 
 * @param matrix - Input matrix
 * @returns Transposed matrix
 * 
 * @example
 * ```typescript
 * const A = [[1, 2], [3, 4]];
 * const result = transpose(A); // [[1, 3], [2, 4]]
 * ```
 */
export function transpose(matrix: Matrix): Matrix {
  if (matrix.length === 0 || matrix[0].length === 0) {
    throw new TypeError('Cannot transpose empty matrix');
  }
  
  return matrix[0].map((_, colIndex) => 
    matrix.map(row => row[colIndex] ?? EPS)
  );
}

// ============================================================================
// ADVANCED MAX-PLUS OPERATIONS
// ============================================================================

/**
 * Matrix power in max-plus algebra: A^n = A ⊗ A ⊗ ... ⊗ A (n times)
 * 
 * @param A - Base matrix (must be square)
 * @param n - Power (non-negative integer)
 * @returns A^n
 * 
 * @example
 * ```typescript
 * const A = [[1, 2], [3, 4]];
 * const result = powOtimes(A, 3); // A ⊗ A ⊗ A
 * ```
 */
export function powOtimes(A: Matrix, n: number): Matrix {
  let result = A;
  for (let i = 0; i < n - 1; i++) {
    result = otimes(result, A);
  }
  return result;
}

/**
 * Trace of a matrix in max-plus algebra: max of diagonal elements
 * 
 * @param A - Square matrix
 * @returns Maximum value on the diagonal
 * 
 * @example
 * ```typescript
 * const A = [[1, 2], [3, 4]];
 * const result = trace(A); // 4 (max of [1, 4])
 * ```
 */
export function trace(A: Matrix): number {
  if (A.length === 0) {
    throw new TypeError('Cannot compute trace of empty matrix');
  }
  
  const diagonal: number[] = [];
  for (let i = 0; i < A.length; i++) {
    const val = A[i]?.[i] ?? EPS;
    diagonal.push(val);
  }
  return Math.max(...diagonal);
}

/**
 * Kleene star (A*): A* = E ⊕ A ⊕ A² ⊕ A³ ⊕ ... ⊕ A^n
 * Represents the transitive closure of a matrix
 * 
 * @param A - Square matrix
 * @returns A*
 * 
 * @example
 * ```typescript
 * const A = [[0, 1], [EPS, 0]];
 * const result = star(A); // Transitive closure
 * ```
 */
export function star(A: Matrix): Matrix {
  if (A.length === 0) {
    throw new TypeError('Cannot compute star of empty matrix');
  }
  
  const n = A.length;
  // Initialize result with E (identity matrix)
  let result: Matrix = [];
  for (let i = 0; i < n; i++) {
    result[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        result[i][j] = 0;
      } else {
        result[i][j] = EPS;
      }
    }
  }

  for (let i = 1; i < n; i++) {
    result = oplus(result, powOtimes(A, i));
  }
  
  return result;
}

/**
 * Plus operation (A+): A+ = A ⊕ A² ⊕ A³ ⊕ ... ⊕ A^n
 * 
 * @param A - Square matrix
 * @returns A+
 * 
 * @example
 * ```typescript
 * const A = [[0, 1], [EPS, 0]];
 * const result = plus(A); // A+ = A ⊕ A² ⊕ A³ ⊕ ...
 * ```
 */
export function plus(A: Matrix): Matrix {
  let result = A;
  const n = A.length;
  
  for (let i = 1; i <= n; i++) {
    result = oplus(result, powOtimes(A, i));
  }
  
  return result;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Create a matrix filled with epsilon values
 * 
 * @param rows - Number of rows
 * @param cols - Number of columns
 * @returns Matrix filled with EPS
 */
export function createEpsilonMatrix(rows: number, cols: number): Matrix {
  const result: Matrix = [];
  for (let i = 0; i < rows; i++) {
    result[i] = new Array(cols).fill(EPS);
  }
  return result;
}

/**
 * Create an identity matrix in max-plus algebra (diagonal elements are 0, others are EPS)
 * 
 * @param size - Size of the square matrix
 * @returns Identity matrix
 */
export function createIdentityMatrix(size: number): Matrix {
  const result: Matrix = [];
  for (let i = 0; i < size; i++) {
    result[i] = [];
    for (let j = 0; j < size; j++) {
      result[i][j] = (i === j) ? 0 : EPS;
    }
  }
  return result;
}

/**
 * Reset a matrix to epsilon values (preserving dimensions)
 * 
 * @param matrix - Matrix to reset
 * @returns Matrix filled with EPS
 */
export function reset(matrix: Matrix): Matrix {
  return matrix.map(row => row.map(() => EPS));
}

/**
 * Check if a matrix is square
 * 
 * @param matrix - Matrix to check
 * @returns True if square, false otherwise
 */
export function isSquare(matrix: Matrix): boolean {
  return matrix.length > 0 && matrix[0] && matrix.length === matrix[0].length;
}

/**
 * Get matrix dimensions
 * 
 * @param matrix - Input matrix
 * @returns Object with rows and cols properties
 */
export function getDimensions(matrix: Matrix): { rows: number; cols: number } {
  return {
    rows: matrix.length,
    cols: matrix.length > 0 && matrix[0] ? matrix[0].length : 0
  };
}
