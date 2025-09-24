# NumPy Cheatsheet

## Import NumPy
```python
import numpy as np
```

## Array Creation

### Basic Array Creation
```python
# From lists
arr = np.array([1, 2, 3, 4])          # 1D array
arr2d = np.array([[1, 2], [3, 4]])    # 2D array

# Data type specification
arr = np.array([1, 2, 3], dtype=np.float64)
```

### Built-in Array Creation Functions
```python
# Zeros, ones, empty
np.zeros(5)                    # [0. 0. 0. 0. 0.]
np.zeros((3, 4))              # 3x4 array of zeros
np.ones((2, 3))               # 2x3 array of ones
np.empty((2, 2))              # Uninitialized 2x2 array

# Identity matrix
np.eye(3)                     # 3x3 identity matrix
np.identity(4)                # 4x4 identity matrix

# Range arrays
np.arange(10)                 # [0 1 2 3 4 5 6 7 8 9]
np.arange(1, 11, 2)          # [1 3 5 7 9]
np.linspace(0, 1, 5)         # [0.   0.25 0.5  0.75 1.  ]

# Random arrays
np.random.random((2, 3))      # Random floats [0, 1)
np.random.randint(0, 10, 5)   # Random integers
np.random.normal(0, 1, (3, 3)) # Normal distribution
np.random.seed(42)            # Set random seed
```

### Special Arrays
```python
np.full((2, 3), 7)            # Fill with specific value
np.full_like(arr, 5)          # Same shape as arr, filled with 5
np.zeros_like(arr)            # Same shape as arr, filled with zeros
np.ones_like(arr)             # Same shape as arr, filled with ones
```

## Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape                     # (2, 3) - dimensions
arr.size                      # 6 - total number of elements
arr.ndim                      # 2 - number of dimensions
arr.dtype                     # data type
arr.itemsize                  # size of each element in bytes
arr.nbytes                    # total bytes consumed
```

## Array Indexing and Slicing

### Basic Indexing
```python
arr = np.array([1, 2, 3, 4, 5])
arr[0]                        # First element: 1
arr[-1]                       # Last element: 5
arr[1:4]                      # Elements 1 to 3: [2 3 4]
arr[::2]                      # Every 2nd element: [1 3 5]
```

### 2D Array Indexing
```python
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[0]                      # First row: [1 2 3]
arr2d[0, 1]                   # Row 0, Col 1: 2
arr2d[:, 1]                   # All rows, Col 1: [2 5 8]
arr2d[1:]                     # All rows from index 1
arr2d[:2, 1:]                 # First 2 rows, cols from index 1
```

### Boolean Indexing
```python
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3                # Boolean array: [False False False True True]
arr[mask]                     # Elements > 3: [4 5]
arr[arr > 3]                  # Direct boolean indexing: [4 5]
```

### Fancy Indexing
```python
arr = np.array([10, 20, 30, 40, 50])
indices = [1, 3, 4]
arr[indices]                  # [20 40 50]
```

## Array Operations

### Arithmetic Operations
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Element-wise operations
arr1 + arr2                   # [5 7 9]
arr1 - arr2                   # [-3 -3 -3]
arr1 * arr2                   # [4 10 18]
arr1 / arr2                   # [0.25 0.4  0.5]
arr1 ** 2                     # [1 4 9]

# Scalar operations
arr1 + 10                     # [11 12 13]
arr1 * 2                      # [2 4 6]
```

### Mathematical Functions
```python
arr = np.array([1, 4, 9, 16])

# Basic math
np.sqrt(arr)                  # Square root
np.exp(arr)                   # Exponential
np.log(arr)                   # Natural logarithm
np.log10(arr)                 # Base 10 logarithm

# Trigonometric
np.sin(arr)
np.cos(arr)
np.tan(arr)

# Rounding
arr = np.array([1.2, 2.7, 3.1])
np.round(arr)                 # [1. 3. 3.]
np.floor(arr)                 # [1. 2. 3.]
np.ceil(arr)                  # [2. 3. 4.]
```

## Array Manipulation

### Reshaping
```python
arr = np.arange(12)
arr.reshape(3, 4)             # 3x4 array
arr.reshape(-1, 3)            # Auto-calculate rows for 3 columns
arr.flatten()                 # 1D array (copy)
arr.ravel()                   # 1D array (view if possible)
```

### Joining Arrays
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

np.concatenate([arr1, arr2])  # [1 2 3 4 5 6]
np.vstack([arr1, arr2])       # Vertical stack (rows)
np.hstack([arr1, arr2])       # Horizontal stack (columns)
np.stack([arr1, arr2])        # Stack along new axis
```

### Splitting Arrays
```python
arr = np.arange(9).reshape(3, 3)
np.hsplit(arr, 3)             # Split horizontally
np.vsplit(arr, 3)             # Split vertically
np.split(arr, 3, axis=0)      # Split along axis 0
```

### Transposing
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.T                         # Transpose: [[1 4], [2 5], [3 6]]
arr.transpose()               # Same as .T
np.transpose(arr)             # Function form
```

## Statistical Operations

### Basic Statistics
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Aggregation functions
arr.sum()                     # Sum of all elements: 21
arr.mean()                    # Mean: 3.5
arr.std()                     # Standard deviation
arr.var()                     # Variance
arr.min()                     # Minimum: 1
arr.max()                     # Maximum: 6

# Along specific axis
arr.sum(axis=0)               # Sum along rows: [5 7 9]
arr.sum(axis=1)               # Sum along columns: [6 15]
arr.mean(axis=0)              # Mean along rows: [2.5 3.5 4.5]
```

### Other Statistical Functions
```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

np.median(arr)                # Median value
np.percentile(arr, 50)        # 50th percentile (median)
np.argmin(arr)                # Index of minimum value
np.argmax(arr)                # Index of maximum value
```

## Linear Algebra

```python
# Matrix multiplication
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.dot(a, b)                  # Matrix multiplication
a @ b                         # Alternative syntax (Python 3.5+)
a.dot(b)                      # Method form

# Linear algebra functions
np.linalg.det(a)              # Determinant
np.linalg.inv(a)              # Inverse matrix
np.linalg.eig(a)              # Eigenvalues and eigenvectors
np.linalg.svd(a)              # Singular value decomposition
```

## Broadcasting

```python
# Arrays with different shapes can be combined
arr = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
scalar = 10                                # ()
arr + scalar                              # Add scalar to all elements

# Broadcasting with different shaped arrays
arr1 = np.array([[1], [2], [3]])         # (3, 1)
arr2 = np.array([10, 20, 30])            # (3,)
arr1 + arr2                               # Result: (3, 3)
```

## Array Comparison and Logic

### Comparison Operations
```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([1, 3, 2, 4])

arr1 == arr2                  # Element-wise equality
arr1 > arr2                   # Element-wise greater than
arr1 >= 2                     # Compare with scalar

# Array-wide comparisons
np.array_equal(arr1, arr2)    # True if arrays are identical
np.allclose(arr1, arr2)       # True if arrays are close (handles floating point)
```

### Logical Operations
```python
arr = np.array([1, 2, 3, 4, 5])

np.logical_and(arr > 1, arr < 5)  # [False True True True False]
np.logical_or(arr < 2, arr > 4)   # [True False False False True]
np.logical_not(arr > 3)           # [True True True False False]

# any() and all()
np.any(arr > 3)               # True (at least one element > 3)
np.all(arr > 0)               # True (all elements > 0)
```

## Sorting and Searching

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# Sorting
np.sort(arr)                  # Return sorted array
arr.sort()                    # Sort in place
np.argsort(arr)               # Indices that would sort the array

# Searching
np.where(arr > 3)             # Indices where condition is True
np.where(arr > 3, arr, 0)     # Replace elements: if >3 keep, else 0
np.searchsorted(sorted_arr, 5) # Index where 5 should be inserted
```

## Advanced Operations

### Unique Values
```python
arr = np.array([1, 2, 2, 3, 3, 3, 4])
np.unique(arr)                # [1 2 3 4]
np.unique(arr, return_counts=True)  # (array([1, 2, 3, 4]), array([1, 2, 3, 1]))
```

### Set Operations
```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

np.intersect1d(arr1, arr2)    # Intersection: [3 4]
np.union1d(arr1, arr2)        # Union: [1 2 3 4 5 6]
np.setdiff1d(arr1, arr2)      # Elements in arr1 not in arr2: [1 2]
```

### Masking and Filtering
```python
arr = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False, True, False, True])

arr[mask]                     # [1 3 5]
np.ma.masked_where(arr > 3, arr)  # Masked array (values > 3 are masked)
```

## File I/O

```python
# Save and load arrays
arr = np.array([1, 2, 3, 4, 5])

# Binary format (faster, smaller)
np.save('array.npy', arr)
loaded_arr = np.load('array.npy')

# Text format (human readable)
np.savetxt('array.txt', arr)
loaded_arr = np.loadtxt('array.txt')

# Multiple arrays
np.savez('arrays.npz', array1=arr1, array2=arr2)
loaded = np.load('arrays.npz')
arr1_loaded = loaded['array1']
```

## Data Types

```python
# Common data types
np.int32, np.int64             # Integers
np.float32, np.float64         # Floating point
np.bool_                       # Boolean
np.complex64, np.complex128    # Complex numbers

# Type conversion
arr = np.array([1, 2, 3])
arr.astype(np.float64)         # Convert to float64
arr.astype('f')                # Convert to float32
```

## Memory and Performance

```python
# Check memory usage
arr.nbytes                     # Total bytes

# Copy vs View
arr_copy = arr.copy()          # Creates a copy
arr_view = arr.view()          # Creates a view (shares data)

# Check if array owns its data
arr.flags.owndata              # True if array owns the data
```

## Common Patterns

### Generate coordinate arrays
```python
x = np.linspace(-5, 5, 11)
y = np.linspace(-5, 5, 11)
X, Y = np.meshgrid(x, y)       # 2D coordinate arrays
```

### Apply function to array
```python
arr = np.array([1, 2, 3, 4])
np.vectorize(lambda x: x**2)(arr)  # Apply function element-wise
```

### Find indices of elements
```python
arr = np.array([1, 3, 2, 4, 2])
indices = np.where(arr == 2)   # Indices where value equals 2
```

This NumPy cheatsheet covers the most essential operations for scientific computing!