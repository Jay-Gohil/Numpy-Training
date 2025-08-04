# NumPy Workshop: Hands-On Code, Exercises & Answers

## Overview

This workshop provides a comprehensive introduction to NumPy, the fundamental package for numerical computing in Python. Through hands-on coding exercises, you'll learn to work with arrays, perform vectorized operations, and apply NumPy to real-world data analysis problems.

*Examples and exercises are adapted from Jake VanderPlas's **Python Data Science Handbook** (Ch. 2: Introduction to NumPy), available at https://jakevdp.github.io/PythonDataScienceHandbook/*

## Prerequisites

- Basic Python programming knowledge (variables, functions, loops)
- Familiarity with Python lists and basic data types
- A Python environment (Jupyter, IPython, or Python interpreter)

## Workshop Schedule

| Time | Topic | Duration   |
|------|-------|----------|
| 09:00 – 09:15 | Welcome and icebreaker | 15 min |
| 09:15 – 09:45 | Why Numpy? | 30 min |
| 09:45 – 10:30 | Creating ndarrays | 45 min |
| 10:30 – 10:45 | *Break* | 15 min |
| 10:45 – 12:15 | Indexing, slicing and views | 90 min |
| 12:15 – 13:15 | *Lunch break* | 60 min |
| 13:15 - 14:45 | Universal functions and aggregations | 90 min |
| 14:45 - 15:00 | *Break* | 15 min |
| 15:00 - 16:00 | Advanced indexing and reshaping | 60 min |
| 16:00 - 16:50 | Performance tuning and mini‑project | 50 min |
| 16:50 - 17:00 | Wrap-up | 10 min |

## How to Use This Document

Each section contains:
- **Questions**: Key learning questions to guide your understanding
- **Objectives**: What you'll be able to do after completing the section
- **Hands-On Code**: Minimal demos to run and explore
- **Exercises**: Short tasks to try individually or in pairs
- **Answers**: Expected results and complete solutions
- **Key Points**: Summary of essential concepts

---

---

## Document Overview

- **Introduction**: covers NumPy basics  
- **Exercises**: a set of practice problems  
- **Solutions**: example answers and explanations  



## 0. Welcome and icebreaker (09:00–09:15)

### Questions
- Is my Python environment properly configured for NumPy?
- How do I verify my NumPy installation and version?

### Objectives
After completing this setup, learners will be able to:
- Verify their Python and NumPy installation
- Import NumPy with the standard convention
- Run basic array operations to confirm the environment works

**Hands‑On Code**  
```python
import sys, numpy as np
print("Python version:", sys.version.split()[0])
print("NumPy version:", np.__version__)
# Check basic array
a = np.array([1,2,3])
print("Test array:", a)
```

**Exercise**  
Run the code above. Confirm that you’re running Python 3.6+ and NumPy 1.15+ (or the latest installed), and that the test array prints correctly.  

**Answer**  
```text
Python version: 3.8.10
NumPy version: 1.21.4
Test array: [1 2 3]
```  
Participants should see their local versions, confirming a valid environment.

### Key Points
- Use `import numpy as np` as the standard NumPy import convention
- Verify your environment with version checks before starting numerical work
- NumPy arrays are the foundation for all scientific Python computing

---

## 1. Why Numpy? (09:15–09:45)

### Questions
- Why should I use NumPy instead of Python lists?
- How much faster is NumPy for numerical computations?
- What makes NumPy arrays more memory efficient?

### Objectives
After completing this episode, learners will be able to:
- Compare performance between Python lists and NumPy arrays
- Explain the memory advantages of NumPy arrays  
- Demonstrate vectorized operations vs explicit loops
- Import NumPy using the standard convention

### What is NumPy?

NumPy (Numerical Python) is the fundamental library for scientific computing in Python. Before diving into comparisons, let's understand what NumPy provides:

```python
import numpy as np

# Your first NumPy array
my_array = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {my_array}")
print(f"Type: {type(my_array)}")
```

NumPy arrays are different from Python lists - they store elements of the same type and enable mathematical operations on entire datasets.

### Basic Array Operations

```python
# Mathematical operations work on entire arrays
numbers = np.array([1, 2, 3, 4])
doubled = numbers * 2
print(f"Original: {numbers}")
print(f"Doubled: {doubled}")

# Element-wise operations happen automatically
result = numbers + 10
print(f"Added 10: {result}")
```

### Performance Comparison

Now that we understand basic NumPy usage, let's see why it's faster:

```python
import timeit
import numpy as np

# Python list approach
def py_sum(n):
    numbers = list(range(n))
    return [i * 2 for i in numbers]

# NumPy approach  
def np_sum(n):
    numbers = np.arange(n)
    return numbers * 2

# Timing comparison
n = 1_000_000
print("Performance Comparison for n =", n)
py_time = timeit.timeit(lambda: py_sum(n), number=10)
np_time = timeit.timeit(lambda: np_sum(n), number=10)

print(f"Python list: {py_time:.4f}s")
print(f"NumPy array: {np_time:.4f}s")
print(f"NumPy is {py_time/np_time:.1f}x faster")
```

### Memory Efficiency

```python
# Memory usage comparison
py_result = py_sum(1000)
np_result = np_sum(1000)

print(f"\nMemory usage comparison:")
print(f"Python list memory: ~{1000 * 28} bytes")  # Rough estimate
print(f"NumPy array memory: {np_result.nbytes} bytes")
print(f"Memory savings: {(1000 * 28 - np_result.nbytes) / (1000 * 28) * 100:.1f}%")
```

### Type Consistency

```python
# Data type advantages
print(f"\nType consistency:")
print(f"Python list element types: {[type(x) for x in py_result[:3]]}")
print(f"NumPy array dtype: {np_result.dtype}")
```

### Vectorized Operations

```python
# Mathematical operations on large datasets
data = np.random.randn(1_000_000)
print(f"\nMath operations on large arrays:")
print(f"Mean: {data.mean():.4f}")
print(f"Standard deviation: {data.std():.4f}")
print(f"Maximum: {data.max():.4f}")
```

### Exercise
1. Time both functions (py_sum vs. np_sum) for n=2_000_000 and record the results
2. Create a function that computes the dot product of two vectors using pure Python vs NumPy
3. Compare the performance of element-wise square root calculation for 1 million numbers

#### Answer

```python
# 1. Timing for larger n
n = 2_000_000
py_time = timeit.timeit(lambda: py_sum(n), number=5)
np_time = timeit.timeit(lambda: np_sum(n), number=5)
print(f"Python list (n=2M): {py_time:.4f}s")
print(f"NumPy array (n=2M): {np_time:.4f}s")
print(f"NumPy is {py_time/np_time:.1f}x faster")

# 2. Dot product comparison
def py_dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def np_dot_product(a, b):
    return np.dot(a, b)

# Test vectors
vec_size = 10_000
a_py = list(range(vec_size))
b_py = list(range(vec_size, 2*vec_size))
a_np = np.array(a_py)
b_np = np.array(b_py)

py_dot_time = timeit.timeit(lambda: py_dot_product(a_py, b_py), number=100)
np_dot_time = timeit.timeit(lambda: np_dot_product(a_np, b_np), number=100)
print(f"Python dot product: {py_dot_time:.4f}s")
print(f"NumPy dot product: {np_dot_time:.4f}s")
print(f"NumPy is {py_dot_time/np_dot_time:.1f}x faster")

# 3. Square root comparison
import math

def py_sqrt(data):
    return [math.sqrt(x) for x in data]

def np_sqrt(data):
    return np.sqrt(data)

test_data = list(range(1, 1_000_001))  # 1 to 1 million
np_test_data = np.array(test_data)

py_sqrt_time = timeit.timeit(lambda: py_sqrt(test_data), number=3)
np_sqrt_time = timeit.timeit(lambda: np_sqrt(np_test_data), number=3)
print(f"Python sqrt: {py_sqrt_time:.4f}s")
print(f"NumPy sqrt: {np_sqrt_time:.4f}s")
print(f"NumPy is {py_sqrt_time/np_sqrt_time:.1f}x faster")
```

### Key Points
- NumPy arrays are significantly faster than Python lists for numerical operations
- NumPy provides vectorized operations that eliminate the need for explicit loops
- Memory usage is much more efficient with NumPy due to homogeneous data types
- NumPy operations are implemented in C, providing near-native performance
- Use `import numpy as np` as the standard convention

---

## 2. Creating ndarrays (09:45–10:30)

### Questions
- How do I create NumPy arrays with different initialization patterns?
- What's the difference between np.zeros, np.ones, np.empty, and np.full?
- How do I inspect array properties like shape, dtype, and memory usage?

### Objectives
After completing this episode, learners will be able to:
- Create arrays using various NumPy creation functions
- Explain the differences between different array initialization methods
- Inspect array attributes and understand their memory implications
- Choose the appropriate creation method for different use cases

### Basic Array Creation Functions

```python
import numpy as np

# Creating arrays from Python lists
list_array = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"From list: {list_array}")
print(f"2D array:\n{matrix}")
```

### Arrays with Specific Values

```python
# np.zeros() - Creates array filled with zeros
zeros_1d = np.zeros(5)                    # 1D array of 5 zeros
zeros_2d = np.zeros((3, 4))               # 3x4 array of zeros  
zeros_int = np.zeros((2, 3), dtype=int)   # Integer zeros

print("np.zeros() - Creates arrays filled with zeros")
print(f"  1D zeros: {zeros_1d}")
print(f"  2D zeros shape: {zeros_2d.shape}")
print(f"  Integer zeros:\n{zeros_int}")

# np.ones() - Creates array filled with ones
ones_2d = np.ones((2, 3))
ones_bool = np.ones(4, dtype=bool)

print(f"\nnp.ones() - Creates arrays filled with ones")
print(f"  2D ones:\n{ones_2d}")
print(f"  Boolean ones: {ones_bool}")

# np.full() - Creates array filled with specified value
full_sevens = np.full((3, 3), 7)
full_pi = np.full(5, np.pi)

print(f"\nnp.full() - Creates arrays filled with any value")
print(f"  Filled with 7:\n{full_sevens}")
print(f"  Filled with π: {full_pi}")
```

### Sequence Generation

```python
# np.arange() - Creates sequences (like Python's range)
arange_basic = np.arange(5)               # 0 to 4
arange_range = np.arange(2, 10, 2)        # 2, 4, 6, 8
arange_float = np.arange(0.5, 3.0, 0.5)   # Works with floats

print("np.arange() - Like range() but returns NumPy array")
print(f"  arange(5): {arange_basic}")
print(f"  arange(2, 10, 2): {arange_range}")  
print(f"  arange(0.5, 3.0, 0.5): {arange_float}")

# np.linspace() - Creates evenly spaced numbers
linspace_basic = np.linspace(0, 1, 5)     # 5 points from 0 to 1
linspace_exclusive = np.linspace(0, 1, 5, endpoint=False)

print(f"\nnp.linspace() - Creates evenly spaced numbers")
print(f"  linspace(0, 1, 5): {linspace_basic}")
print(f"  without endpoint: {linspace_exclusive}")
```

### Special Arrays

```python
# Identity matrices
identity_3x3 = np.eye(3)                  # 3x3 identity matrix
print("np.eye() - Creates identity matrices")
print(f"  3x3 identity:\n{identity_3x3}")

# Random arrays
np.random.seed(42)  # For reproducible results
random_uniform = np.random.random((2, 3))    # Uniform [0, 1)
random_normal = np.random.normal(0, 1, (2, 3))  # Normal distribution

print(f"\nRandom arrays:")
print(f"  Uniform random:\n{random_uniform}")
print(f"  Normal distribution:\n{random_normal}")
```

### Array Inspection

```python
# Create sample array for inspection
sample_array = np.random.normal(0, 1, (3, 4)).astype(np.float32)

print("Array inspection:")
print(f"  Array:\n{sample_array}")
print(f"  .shape: {sample_array.shape} - Dimensions")
print(f"  .dtype: {sample_array.dtype} - Data type")  
print(f"  .ndim: {sample_array.ndim} - Number of dimensions")
print(f"  .size: {sample_array.size} - Total elements")
print(f"  .nbytes: {sample_array.nbytes} - Total bytes")
print(f"  .itemsize: {sample_array.itemsize} - Bytes per element")
```

### Data Type Information

```python
# Understanding data types
int_array = np.array([1, 2, 3, 4])
float_array = np.array([1.0, 2.5, 3.7])

print(f"\nData types:")
print(f"Integer array dtype: {int_array.dtype}")
print(f"Float array dtype: {float_array.dtype}")

# Memory comparison
data = [1, 2, 3, 4, 5]
small_ints = np.array(data, dtype=np.int8)
large_ints = np.array(data, dtype=np.int64)

print(f"\nMemory efficiency:")
print(f"int8 memory: {small_ints.nbytes} bytes")
print(f"int64 memory: {large_ints.nbytes} bytes")
```

### Exercise
1. Create a 3×4 array filled with the value 7 using np.full, then report min, max, and mean
2. Create a 5×5 identity matrix and modify it to have 2s on the diagonal  
3. Generate a 4×4 array where each element equals i*j (row × column indices)
4. Create arrays with different data types and compare their memory usage per element

#### Answer

```python
# 1. Array filled with 7 using np.full()
sevens_array = np.full((3, 4), 7)
print(f"Array filled with 7s:\n{sevens_array}")
print(f"Min: {sevens_array.min()}, Max: {sevens_array.max()}, Mean: {sevens_array.mean()}")

# 2. Modified identity matrix
identity = np.eye(5)
print(f"Original identity matrix:\n{identity}")

# Method 1: In-place multiplication
identity *= 2  # Scale diagonal by 2
print(f"2x Identity matrix:\n{identity}")

# Alternative method: Direct creation
identity2 = np.eye(5) * 2
print(f"Alternative approach:\n{identity2}")

# 3. Array where each element equals i*j (row × column)
# Method 1: Using np.fromfunction() - elegant approach
def index_product(i, j):
    return i * j

product_array = np.fromfunction(index_product, (4, 4))
print(f"i*j array using fromfunction:\n{product_array}")

# Method 2: Using broadcasting - more explicit
i = np.arange(4).reshape(4, 1)  # Column vector [0, 1, 2, 3]
j = np.arange(4)                # Row vector [0, 1, 2, 3]
product_array2 = i * j          # Broadcasting creates 4x4 result
print(f"i*j array using broadcasting:\n{product_array2}")

# 4. Data type memory comparison
print(f"\nData type memory usage comparison:")
test_data = [1, 2, 3, 4, 5]
dtypes = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]

print(f"{'Data Type':<10} {'Bytes/Element':<13} {'Total Bytes':<11}")
print("-" * 35)

for dtype in dtypes:
    arr = np.array(test_data, dtype=dtype)
    print(f"{dtype.__name__:<10} {arr.itemsize:<13} {arr.nbytes:<11}")

# Memory efficiency demonstration
large_int8 = np.ones(1_000_000, dtype=np.int8)
large_int64 = np.ones(1_000_000, dtype=np.int64)
print(f"\nMemory efficiency (1M elements):")
print(f"int8: {large_int8.nbytes / 1024**2:.1f} MB")
print(f"int64: {large_int64.nbytes / 1024**2:.1f} MB")
print(f"Savings: {(1 - large_int8.nbytes/large_int64.nbytes)*100:.1f}%")
```

### Key Points
- `np.zeros()` creates arrays filled with zeros - most common initialization
- `np.ones()` creates arrays filled with ones - useful for probabilities and weights
- `np.full(shape, value)` fills arrays with any constant value
- `np.arange()` is like range() but returns NumPy arrays and works with floats
- `np.linspace()` creates evenly spaced numbers - better for mathematical intervals
- `np.eye()` creates identity matrices for linear algebra operations
- Array properties: `.shape`, `.dtype`, `.ndim`, `.size`, `.nbytes`, `.itemsize`
- Choose appropriate data types to minimize memory usage

## 3. Indexing, slicing and views (10:45–12:15)

### Questions
- How do I access and modify specific elements or sections of arrays?
- What's the difference between views and copies when slicing arrays?
- How can I use boolean and fancy indexing for data filtering and selection?

### Objectives
After completing this episode, learners will be able to:
- Use basic indexing and slicing to access array elements and subarrays
- Apply boolean indexing for conditional data selection
- Understand the difference between views and copies in array operations
- Use fancy indexing for complex array manipulations

**Hands‑On Code**  
```python
import numpy as np
x = np.arange(12).reshape(3,4)
print("Original array:")
print(x)

# Basic indexing
print(f"\nBasic indexing:")
print(f"x[0,1] = {x[0,1]}")  # single element
print(f"x[2,3] = {x[2,3]}")  # single element
print(f"x[-1,-1] = {x[-1,-1]}")  # negative indices

# Slicing
print(f"\nSlicing:")
print(f"First row: {x[0, :]}")
print(f"Last column: {x[:, -1]}")
print(f"First 2 rows, last 2 columns:\n{x[:2, -2:]}")
print(f"Every other row: \n{x[::2, :]}")
print(f"Reverse rows:\n{x[::-1, :]}")

# Integer array indexing (fancy indexing)
rows = np.array([0, 2])
cols = np.array([1, 3])
print(f"\nFancy indexing:")
print(f"x[{rows}, {cols}] = {x[rows, cols]}")  # picks (0,1) and (2,3)

# More complex fancy indexing
print(f"Multiple row selection:\n{x[[0, 2], :]}")  # Select rows 0 and 2
print(f"Multiple col selection:\n{x[:, [1, 3]]}")  # Select columns 1 and 3

# Boolean indexing
mask = x % 2 == 0
print(f"\nBoolean indexing:")
print(f"Even mask:\n{mask}")
evens = x[mask]
print("Even values:", evens)

# Combined boolean and integer indexing
print(f"Even values in first two rows: {x[:2][x[:2] % 2 == 0]}")

# Advanced boolean operations
print(f"Values > 5 and < 10: {x[(x > 5) & (x < 10)]}")
print(f"Values <= 3 or >= 9: {x[(x <= 3) | (x >= 9)]}")

# Modifying arrays through indexing
print(f"\nModifying arrays:")
# Slice view and modify
sub = x[1:3, 2:4]
print(f"Subarray (view):\n{sub}")
sub[:] = 100
print(f"After modifying view, original x:\n{x}")

# Reset array for next examples
x = np.arange(12).reshape(3,4)

# Copy vs view demonstration
print(f"\nCopy vs View:")
cp = x[:, :2].copy()  # This is a copy
view = x[:, :2]       # This is a view

cp[:] = -1
view[:] = 999

print("After modifying copy and view:")
print(f"Original x:\n{x}")
print(f"Copy (independent):\n{cp}")

# Advanced indexing patterns
x = np.arange(24).reshape(4, 6)
print(f"\nAdvanced patterns on 4x6 array:")
print(f"Original:\n{x}")

# Diagonal elements
diag_indices = np.arange(min(x.shape))
print(f"Diagonal: {x[diag_indices, diag_indices]}")

# Block selection
print(f"2x2 blocks (top-left and bottom-right):")
print(f"Top-left 2x2:\n{x[:2, :2]}")
print(f"Bottom-right 2x2:\n{x[-2:, -2:]}")

# Conditional replacement
x_copy = x.copy()
x_copy[x_copy > 15] = -1
print(f"Replace values > 15 with -1:\n{x_copy}")
```

**Exercise**  
1. Given `x = np.arange(16).reshape(4,4)`, use boolean indexing to extract all elements divisible by 3.
2. Use integer indexing to select the diagonal elements.
3. Create a checkerboard pattern of True/False values using boolean indexing.
4. Extract the four corner elements of the array using fancy indexing.

**Answer**  
```python
# Setup
x = np.arange(16).reshape(4,4)
print("Original array:")
print(x)

# 1. Elements divisible by 3
divisible_by_3 = x[x % 3 == 0]
print(f"Divisible by 3: {divisible_by_3}")  # [0 3 6 9 12 15]

# 2. Diagonal elements
diag = x[np.arange(4), np.arange(4)]
print(f"Diagonal elements: {diag}")  # [0 5 10 15]

# Alternative using np.diag
diag_alt = np.diag(x)
print(f"Diagonal (alternative): {diag_alt}")

# 3. Checkerboard pattern
rows, cols = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')
checkerboard = (rows + cols) % 2 == 0
print(f"Checkerboard pattern:\n{checkerboard}")

# Apply checkerboard pattern
checkerboard_values = x[checkerboard]
print(f"Checkerboard values: {checkerboard_values}")

# 4. Four corner elements
corners = x[[0, 0, -1, -1], [0, -1, 0, -1]]
print(f"Corner elements: {corners}")  # [0, 3, 12, 15]

# Alternative approach for corners
corner_coords = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
corners_alt = [x[r, c] for r, c in corner_coords]
print(f"Corners (alternative): {corners_alt}")

# Bonus: Extract anti-diagonal
anti_diag = x[np.arange(4), np.arange(4)[::-1]]
print(f"Anti-diagonal: {anti_diag}")  # [3, 6, 9, 12]
```  

### Key Points
- Use square brackets `[]` for indexing and slicing arrays
- Slicing creates views by default, which share memory with the original array
- Boolean indexing enables powerful conditional data selection
- Fancy indexing with integer arrays allows for complex element selection
- Always be aware of whether operations create views or copies for memory efficiency

---

## 4. Universal functions and aggregations (13:15–14:45)

### Questions
- What are universal functions (ufuncs) and how do they work?
- What is broadcasting and how does it enable operations between arrays of different shapes?
- How do vectorized operations compare to traditional loops in performance?
- How can I apply functions element-wise across arrays efficiently?

### Objectives
After completing this episode, learners will be able to:
- Use NumPy's universal functions for element-wise operations
- Understand NumPy's broadcasting rules and when they apply
- Use broadcasting for efficient array operations without creating intermediate copies
- Compare vectorized operations with explicit loops for performance
- Apply aggregation functions and conditional operations on arrays
- Understand when and how to use vectorized computations

**Hands‑On Code**  
```python
import numpy as np

# Generate sample data
x = np.linspace(0, 2*np.pi, 5)
print(f"Input array: {x}")

# Basic universal functions
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)
sum_trig = np.add(y_sin, y_cos)

print(f"sin(x): {y_sin}")
print(f"cos(x): {y_cos}")
print(f"tan(x): {y_tan}")
print(f"sin + cos: {sum_trig}")

# Mathematical operations
y_exp = np.exp(x)
y_log = np.log(x + 1)  # Add 1 to avoid log(0)
y_sqrt = np.sqrt(x)
y_sq = np.square(x)
y_power = np.power(x, 3)

# Using out parameter to avoid temporaries
out = np.empty_like(x)
np.multiply(x, 2, out=out)
print(f"2x (using out parameter): {out}")

# More ufuncs
print(f"exp(x): {y_exp}")
print(f"log(x+1): {y_log}")
print(f"sqrt(x): {y_sqrt}")
print(f"x^2: {y_sq}")
print(f"x^3: {y_power}")

# Comparison operations (element-wise)
a = np.array([1, 3, 5, 7, 9])
b = np.array([2, 3, 4, 7, 8])
print(f"\nComparisons:")
print(f"a = {a}")
print(f"b = {b}")
print(f"a > b: {a > b}")
print(f"a == b: {a == b}")
print(f"a >= b: {a >= b}")

# Logical operations
print(f"a > 4: {a > 4}")
print(f"b < 5: {b < 5}")
print(f"(a > 4) & (b < 5): {(a > 4) & (b < 5)}")
print(f"(a > 4) | (b < 5): {(a > 4) | (b < 5)}")

# Comprehensive Aggregation Functions (inspired by VanderPlas Ch 2.04)
print(f"\n=== Aggregation Functions ===")

# Performance comparison: Python vs NumPy
big_array = np.random.random(1000000)
print(f"Performance comparison on 1M elements:")

# Time comparison (conceptual - actual timing would vary)
# Python sum: ~100ms, NumPy sum: ~500μs (200x faster)
print(f"Sum result: {np.sum(big_array):.4f}")
print(f"NumPy sum is ~200x faster than Python's built-in sum()")

# Comprehensive aggregation table (following VanderPlas Table 2-3)
data = np.random.randn(1000)
print(f"\nComprehensive aggregations on random data (n=1000):")
print(f"{'Function':<20} {'Result':<15} {'NaN-safe version'}")
print("-" * 55)
print(f"{'np.sum':<20} {np.sum(data):<15.4f} {'np.nansum'}")
print(f"{'np.prod':<20} {np.prod(data[:10]):<15.4f} {'np.nanprod'}")  # Use subset for product
print(f"{'np.mean':<20} {np.mean(data):<15.4f} {'np.nanmean'}")
print(f"{'np.std':<20} {np.std(data):<15.4f} {'np.nanstd'}")
print(f"{'np.var':<20} {np.var(data):<15.4f} {'np.nanvar'}")
print(f"{'np.min':<20} {np.min(data):<15.4f} {'np.nanmin'}")
print(f"{'np.max':<20} {np.max(data):<15.4f} {'np.nanmax'}")
print(f"{'np.argmin':<20} {np.argmin(data):<15} {'np.nanargmin'}")
print(f"{'np.argmax':<20} {np.argmax(data):<15} {'np.nanargmax'}")
print(f"{'np.median':<20} {np.median(data):<15.4f} {'np.nanmedian'}")
print(f"{'np.percentile':<20} {np.percentile(data, 25):<15.4f} {'np.nanpercentile'}")
print(f"{'np.any':<20} {str(np.any(data > 0)):<15} {'N/A'}")
print(f"{'np.all':<20} {str(np.all(data > -5)):<15} {'N/A'}")

# Multi-dimensional aggregations with axis examples
print(f"\n=== Multi-dimensional Aggregations ===")
matrix = np.random.random((3, 4))
print(f"Matrix (3x4):\n{matrix}")
print(f"Matrix shape: {matrix.shape}")

print(f"\nAggregations along different axes:")
print(f"Overall sum: {matrix.sum():.4f}")
print(f"Sum of each column (axis=0): {matrix.sum(axis=0)}")  # Shape: (4,)
print(f"Sum of each row (axis=1): {matrix.sum(axis=1)}")     # Shape: (3,)

print(f"\nAxis specification explanation:")
print(f"axis=0 collapses rows → result has shape {matrix.sum(axis=0).shape}")
print(f"axis=1 collapses columns → result has shape {matrix.sum(axis=1).shape}")

# Method vs function syntax
print(f"\nMethod vs Function syntax (equivalent results):")
print(f"matrix.sum(): {matrix.sum():.4f}")
print(f"np.sum(matrix): {np.sum(matrix):.4f}")
print(f"matrix.min(): {matrix.min():.4f}")
print(f"np.min(matrix): {np.min(matrix):.4f}")

# Practical example: US Presidents Heights (inspired by VanderPlas)
print(f"\n=== Practical Example: Statistical Analysis ===")
# Simulated height data (in cm) - representing US president heights
np.random.seed(42)  # For reproducible results
heights = np.random.normal(179.7, 6.9, 44)  # Mean and std from VanderPlas example
heights = np.round(heights).astype(int)

print(f"Height analysis (n={len(heights)} presidents):")
print(f"Mean height: {heights.mean():.1f} cm")
print(f"Standard deviation: {heights.std():.1f} cm")
print(f"Minimum height: {heights.min()} cm")
print(f"Maximum height: {heights.max()} cm")
print(f"25th percentile: {np.percentile(heights, 25):.1f} cm")
print(f"Median: {np.median(heights):.1f} cm")  
print(f"75th percentile: {np.percentile(heights, 75):.1f} cm")

# Boolean aggregations
tall_presidents = heights > 180
print(f"\nBoolean aggregations:")
print(f"Number of presidents > 180cm: {np.sum(tall_presidents)}")
print(f"Percentage > 180cm: {100 * np.mean(tall_presidents):.1f}%")
print(f"Any president > 190cm? {np.any(heights > 190)}")
print(f"All presidents > 160cm? {np.all(heights > 160)}")

# Working with missing data (NaN-safe functions)
heights_with_missing = heights.astype(float)
heights_with_missing[[5, 12, 23]] = np.nan  # Introduce some missing values

print(f"\n=== Working with Missing Data ===")
print(f"Data with missing values: {np.sum(np.isnan(heights_with_missing))} NaN values")
print(f"Regular mean (with NaN): {np.mean(heights_with_missing)}")  # Returns NaN
print(f"NaN-safe mean: {np.nanmean(heights_with_missing):.1f} cm")
print(f"NaN-safe std: {np.nanstd(heights_with_missing):.1f} cm")
print(f"NaN-safe count: {np.sum(~np.isnan(heights_with_missing))} valid values")

# Cumulative operations
arr = np.array([1, 2, 3, 4, 5])
print(f"\nCumulative operations on {arr}:")
print(f"Cumulative sum: {np.cumsum(arr)}")
print(f"Cumulative product: {np.cumprod(arr)}")

# Advanced ufunc features
# Reduce operations
print(f"\nReduce operations:")
print(f"Sum via reduce: {np.add.reduce(arr)}")
print(f"Product via reduce: {np.multiply.reduce(arr)}")

# Accumulate operations
print(f"Add accumulate: {np.add.accumulate(arr)}")
print(f"Multiply accumulate: {np.multiply.accumulate(arr)}")

# Outer operations
small_arr = np.array([1, 2, 3])
print(f"Outer product:\n{np.multiply.outer(small_arr, small_arr)}")

# Conditional operations
data = np.array([-2, -1, 0, 1, 2])
print(f"\nConditional operations on {data}:")
print(f"Absolute values: {np.abs(data)}")
print(f"Sign function: {np.sign(data)}")
print(f"Where positive, square; else, zero: {np.where(data > 0, data**2, 0)}")

# Combining multiple conditions
result = np.where((data > -1) & (data < 1), data * 10, data)
print(f"Scale values in [-1,1) by 10: {result}")
```

### Exercise
1. For `x = np.array([1, 2, 3, 4])`, compute `ln(x)`, `sqrt(x)`, and cumulative product.
2. Create two arrays and compute element-wise max and min.
3. Use `np.where` to replace negative values with 0 and positive values with their square.
4. Compute the dot product and outer product of two vectors.

### Answer
```python
# 1. Basic operations
x = np.array([1, 2, 3, 4])
ln_x = np.log(x)
sqrt_x = np.sqrt(x)
cumprod_x = np.cumprod(x)

print(f"x = {x}")
print(f"ln(x) = {ln_x}")      # [0. 0.693 1.098 1.386]
print(f"sqrt(x) = {sqrt_x}")  # [1. 1.414 1.732 2.]
print(f"cumprod(x) = {cumprod_x}")  # [1 2 6 24]

# 2. Element-wise max and min
a = np.array([1, 5, 3, 9, 2])
b = np.array([2, 3, 7, 1, 4])
element_max = np.maximum(a, b)
element_min = np.minimum(a, b)

print(f"\na = {a}")
print(f"b = {b}")
print(f"element-wise max: {element_max}")  # [2 5 7 9 4]
print(f"element-wise min: {element_min}")  # [1 3 3 1 2]

# 3. Conditional replacement
data = np.array([-3, -1, 0, 2, 4, -2])
result = np.where(data < 0, 0, data**2)
print(f"\nOriginal: {data}")
print(f"Negative→0, Positive→square: {result}")  # [0 0 0 4 16 0]

# Alternative for more complex conditions:
result2 = np.where(data < 0, 0, np.where(data == 0, 0, data**2))
print(f"Same result with nested where: {result2}")

# 4. Dot and outer products
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

dot_product = np.dot(u, v)
outer_product = np.outer(u, v)

print(f"\nu = {u}")
print(f"v = {v}")
print(f"Dot product: {dot_product}")  # 32 (1*4 + 2*5 + 3*6)
print(f"Outer product:\n{outer_product}")
# [[4  5  6]
#  [8 10 12]
#  [12 15 18]]

# Bonus: Cross product (for 3D vectors)
cross_product = np.cross(u, v)
print(f"Cross product: {cross_product}")  # [-3  6 -3]
```  

### Key Points - Universal Functions
- Universal functions (ufuncs) operate element-wise on arrays and are highly optimized
- Vectorized operations are typically 10-100x faster than explicit Python loops
- NumPy provides comprehensive mathematical, trigonometric, and logical functions
- Use `np.where()` for conditional operations and element-wise logic

---

### Broadcasting

**Hands‑On Code**  
```python
import numpy as np

# Broadcasting basics
print("=== Broadcasting Examples ===")

# 1D to 2D broadcasting
M = np.ones((3,4))
v = np.arange(4)
print(f"Matrix M (3x4):\n{M}")
print(f"Vector v (4,): {v}")
print(f"M + v (broadcasting v to each row):\n{M + v}")

# Broadcasting to columns
w = np.arange(3)
print(f"\nVector w (3,): {w}")
print(f"Broadcasting w to each column:")
print(f"Method 1 - transpose: \n{(M.T + w).T}")
print(f"Method 2 - reshape: \n{M + w[:, None]}")
print(f"Method 3 - newaxis: \n{M + w[:, np.newaxis]}")

# Scalar broadcasting
print(f"\nScalar broadcasting:")
print(f"M + 5:\n{M + 5}")

# Higher-dimensional broadcasting
A = np.arange(8).reshape(2,4)
b = np.array([10, 20])
print(f"\nHigher-dimensional example:")
print(f"Array A (2x4):\n{A}")
print(f"Vector b (2,): {b}")
print(f"A + b[:, None] (add b as columns):\n{A + b[:, None]}")

# Broadcasting rules demonstration
print(f"\n=== Broadcasting Rules (VanderPlas Ch 2.05) ===")

# The Three Rules of Broadcasting:
# Rule 1: If arrays differ in number of dimensions, pad the smaller one with 1s on the left
# Rule 2: If shapes don't match in any dimension, stretch the dimension with size 1
# Rule 3: If sizes disagree and neither is 1, raise an error

print("Broadcasting Rules:")
print("Rule 1: Pad with 1s on left side for fewer dimensions")
print("Rule 2: Stretch size-1 dimensions to match")  
print("Rule 3: Error if sizes disagree and neither is 1")

# Broadcasting examples with detailed rule application
print(f"\n=== Broadcasting Examples with Rule Analysis ===")

# Example 1: 2D + 1D 
print("Example 1: Adding 2D array + 1D array")
M = np.ones((2, 3))
a = np.arange(3)
print(f"M.shape = {M.shape}")
print(f"a.shape = {a.shape}")

print("Rule 1: Pad a's shape with 1s on left → a.shape becomes (1, 3)")
print("Rule 2: First dimension: M(2) vs a(1) → stretch a to (2, 3)")  
print("Rule 3: All dimensions compatible → result shape (2, 3)")

result1 = M + a
print(f"Result:\n{result1}")

# Example 2: Both arrays need broadcasting
print(f"\nExample 2: Both arrays need broadcasting")
a = np.arange(3).reshape((3, 1))  # Shape: (3, 1)
b = np.arange(3)                  # Shape: (3,)
print(f"a.shape = {a.shape}")
print(f"b.shape = {b.shape}")

print("Rule 1: Pad b's shape → b.shape becomes (1, 3)")
print("Rule 2: Dimension 0: a(3) vs b(1) → stretch b to (3, 3)")
print("Rule 2: Dimension 1: a(1) vs b(3) → stretch a to (3, 3)")
print("Final result shape: (3, 3)")

result2 = a + b
print(f"Result:\n{result2}")

# Example 3: Incompatible arrays (demonstrates Rule 3)
print(f"\nExample 3: Incompatible arrays")
M = np.ones((3, 2))
a = np.arange(3)
print(f"M.shape = {M.shape}")
print(f"a.shape = {a.shape}")

print("Rule 1: Pad a's shape → a.shape becomes (1, 3)")
print("Rule 2: Dimension 0: M(3) vs a(1) → stretch a to (3, 3)")
print("Rule 3: Dimension 1: M(2) vs a(3) → ERROR! Neither is 1")

try:
    incompatible = M + a
except ValueError as e:
    print(f"ValueError: {e}")

# Fix the incompatible case
print("Fix: Reshape a to be compatible")
a_fixed = a[:, np.newaxis]  # Shape: (3, 1)
print(f"a_fixed.shape = {a_fixed.shape}")
result3 = M + a_fixed
print(f"M + a_fixed works! Result shape: {result3.shape}")

# Practical broadcasting applications
print(f"\n=== Practical Broadcasting Applications ===")

# Centering an array (VanderPlas example)
print("1. Centering data (zero-mean)")
X = np.random.random((10, 3))  # 10 observations, 3 features
print(f"Data shape: {X.shape}")
print(f"Original means: {X.mean(0)}")

Xmean = X.mean(0)  # Shape: (3,) - mean of each feature
X_centered = X - Xmean  # Broadcasting: (10,3) - (3,) → (10,3)
print(f"Centered means: {X_centered.mean(0)}")  # Should be ~0

# 2D function plotting (VanderPlas example)
print(f"\n2. 2D function evaluation")
x = np.linspace(0, 5, 50)                    # Shape: (50,)
y = np.linspace(0, 5, 50)[:, np.newaxis]    # Shape: (50, 1)
print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")

# Broadcasting creates a 2D grid
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
print(f"Result shape: {z.shape}")  # (50, 50)
print("This creates a 2D function over the entire x-y grid!")

# Distance calculations
print(f"\n3. Efficient distance calculations")
points = np.random.random((5, 2))  # 5 points in 2D
center = np.array([0.5, 0.5])

# Method 1: Broadcasting for distance from center
distances = np.sqrt(np.sum((points - center)**2, axis=1))
print(f"Distances from center: {distances}")

# Method 2: All pairwise distances using broadcasting
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # Shape: (5, 5, 2)
pairwise_dist = np.sqrt(np.sum(diff**2, axis=2))
print(f"Pairwise distance matrix shape: {pairwise_dist.shape}")
```

**Exercise - Broadcasting**  
1. Given `X = np.ones((4,3,2))` and `v = np.array([1,2])`, broadcast `v` to add to the last axis of `X`.
2. Create a 5x5 matrix where each element (i,j) equals the sum i+j using broadcasting.
3. Normalize a 2D array so each column has mean 0 and standard deviation 1.

**Answer - Broadcasting**  
```python
# 1. Broadcasting to last axis
X = np.ones((4,3,2))
v = np.array([1,2])
result = X + v  # v broadcasts to last dimension automatically
print(f"X.shape: {X.shape}")
print(f"v.shape: {v.shape}")
print(f"Result shape: {result.shape}")  # (4,3,2)

# 2. Matrix where element (i,j) = i+j
i_indices = np.arange(5)[:, np.newaxis]  # Shape: (5, 1)
j_indices = np.arange(5)  # Shape: (5,)
sum_matrix = i_indices + j_indices
print(f"Sum matrix (i+j):\n{sum_matrix}")

# 3. Column-wise normalization
data = np.random.randn(6, 4)  # 6 rows, 4 columns
col_means = data.mean(axis=0, keepdims=True)  # Shape: (1, 4)
col_stds = data.std(axis=0, keepdims=True)    # Shape: (1, 4)
normalized = (data - col_means) / col_stds
print(f"Normalized column means: {normalized.mean(axis=0)}")
```  

### Key Points - Broadcasting
- Broadcasting allows operations between arrays of different shapes without explicit loops
- Arrays are compatible for broadcasting when their trailing dimensions are equal or one of them is 1
- Broadcasting is memory-efficient as it doesn't create intermediate copies
- Use `keepdims=True` in reduction operations to maintain dimensions for broadcasting

---

### Aggregations

**Hands‑On Code**  
```python
import numpy as np

# Comprehensive Aggregation Functions (inspired by VanderPlas Ch 2.04)
print(f"=== Aggregation Functions ===")

# Performance comparison: Python vs NumPy
big_array = np.random.random(1000000)
print(f"Performance comparison on 1M elements:")

# Time comparison (conceptual - actual timing would vary)
# Python sum: ~100ms, NumPy sum: ~500μs (200x faster)
print(f"Sum result: {np.sum(big_array):.4f}")
print(f"NumPy sum is ~200x faster than Python's built-in sum()")

# Comprehensive aggregation table (following VanderPlas Table 2-3)
data = np.random.randn(1000)
print(f"\nComprehensive aggregations on random data (n=1000):")
print(f"{'Function':<20} {'Result':<15} {'NaN-safe version'}")
print("-" * 55)
print(f"{'np.sum':<20} {np.sum(data):<15.4f} {'np.nansum'}")
print(f"{'np.mean':<20} {np.mean(data):<15.4f} {'np.nanmean'}")
print(f"{'np.std':<20} {np.std(data):<15.4f} {'np.nanstd'}")
print(f"{'np.min':<20} {np.min(data):<15.4f} {'np.nanmin'}")
print(f"{'np.max':<20} {np.max(data):<15.4f} {'np.nanmax'}")
print(f"{'np.median':<20} {np.median(data):<15.4f} {'np.nanmedian'}")

# Multi-dimensional aggregations with axis examples
print(f"\n=== Multi-dimensional Aggregations ===")
matrix = np.random.random((3, 4))
print(f"Matrix (3x4):\n{matrix}")

print(f"\nAggregations along different axes:")
print(f"Overall sum: {matrix.sum():.4f}")
print(f"Sum of each column (axis=0): {matrix.sum(axis=0)}")  # Shape: (4,)
print(f"Sum of each row (axis=1): {matrix.sum(axis=1)}")     # Shape: (3,)

# Boolean aggregations
heights = np.array([170, 175, 180, 185, 190])
tall_threshold = 180
print(f"\nBoolean aggregations:")
print(f"Heights: {heights}")
print(f"Number > {tall_threshold}cm: {np.sum(heights > tall_threshold)}")
print(f"Percentage > {tall_threshold}cm: {100 * np.mean(heights > tall_threshold):.1f}%")
```

### Key Points - Aggregations
- Aggregation functions like `sum()`, `mean()`, `max()` can operate on entire arrays or along specific axes
- Use `axis` parameter to control which dimension is aggregated
- NaN-safe versions (e.g., `np.nanmean`) handle missing data appropriately
- Boolean aggregations enable counting and percentage calculations

---

## 5. Advanced indexing and reshaping (15:00–16:00)

### Questions
- How can I use advanced indexing techniques for complex data manipulation?
- What's the difference between reshaping and flattening arrays?
- How do I combine and split arrays efficiently?

### Objectives
After completing this episode, learners will be able to:
- Use advanced indexing techniques including fancy and boolean indexing
- Reshape arrays while understanding memory layout implications
- Combine and split arrays using concatenation and stacking
- Apply advanced indexing to real-world data manipulation problems

**Hands‑On Code**  
```python
import numpy as np

# Boolean indexing
print("=== Boolean Indexing ===")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = data > 5
print(f"Data: {data}")
print(f"Mask (data > 5): {mask}")
print(f"Values > 5: {data[mask]}")

# Fancy indexing
print(f"\n=== Fancy Indexing ===")
indices = [1, 3, 7]
print(f"Elements at indices {indices}: {data[indices]}")

# 2D boolean indexing
matrix = np.random.randint(0, 10, (4, 4))
print(f"\nMatrix:\n{matrix}")
print(f"Values > 5:\n{matrix[matrix > 5]}")

# Array reshaping
print(f"\n=== Reshaping Arrays ===")
arr = np.arange(12)
print(f"Original: {arr}")
print(f"Reshaped (3,4):\n{arr.reshape(3, 4)}")
print(f"Reshaped (2,6):\n{arr.reshape(2, 6)}")

# Array concatenation
print(f"\n=== Array Concatenation ===")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Concatenate: {np.concatenate([a, b])}")

# 2D concatenation
matrix1 = np.ones((2, 3))
matrix2 = np.zeros((2, 3))
print(f"Vertical stack:\n{np.vstack([matrix1, matrix2])}")
print(f"Horizontal stack:\n{np.hstack([matrix1, matrix2])}")
```

### Key Points - Advanced Indexing
- Boolean indexing provides powerful filtering capabilities
- Fancy indexing allows selection of arbitrary array elements
- Reshaping changes array dimensions while preserving data
- Use concatenation and stacking to combine arrays

---
- Stack and split arrays using various NumPy functions
- Manipulate array dimensions for different data processing needs

**Hands‑On Code**  
```python
import numpy as np

# Advanced indexing demonstrations
print("=== Advanced Indexing ===")

# Fancy indexing for reordering
x = np.arange(9)
order = [8, 2, 5, 0]
reordered = x[order]
print(f"Original array: {x}")
print(f"Reordered by indices {order}: {reordered}")

# Fancy indexing with 2D arrays
matrix = np.arange(12).reshape(3, 4)
print(f"\nMatrix:\n{matrix}")

# Select specific elements using coordinate arrays
rows = np.array([0, 1, 2])
cols = np.array([1, 2, 3])
elements = matrix[rows, cols]  # Gets (0,1), (1,2), (2,3)
print(f"Elements at (0,1), (1,2), (2,3): {elements}")

# Select multiple rows and columns
selected_rows = matrix[[0, 2]]  # Rows 0 and 2
selected_cols = matrix[:, [1, 3]]  # Columns 1 and 3
selected_block = matrix[[0, 2]][:, [1, 3]]  # Intersection
print(f"Selected rows [0,2]:\n{selected_rows}")
print(f"Selected columns [1,3]:\n{selected_cols}")
print(f"Block [0,2] × [1,3]:\n{selected_block}")

# Boolean indexing for modification
y = np.arange(10)
print(f"\nBefore modification: {y}")
y[y % 2 == 1] *= -1  # Negate odd numbers
print(f"After negating odds: {y}")

# Complex boolean conditions
data = np.random.randn(20)
# Replace values outside [-2, 2] with the median
median_val = np.median(data)
mask = (data < -2) | (data > 2)
data[mask] = median_val
print(f"Data after clipping outliers (replaced {np.sum(mask)} values)")

# Comprehensive Boolean Arrays & Masking (VanderPlas Ch 2.06)
print(f"\n=== Comparison Operators as Universal Functions ===")

# All comparison operators are ufuncs
x = np.array([1, 2, 3, 4, 5])
print(f"Array x: {x}")

# Comparison operators table (following VanderPlas)
print(f"\nComparison Operators:")
print(f"x < 3:  {x < 3}")   # np.less(x, 3)
print(f"x > 3:  {x > 3}")   # np.greater(x, 3)  
print(f"x <= 3: {x <= 3}")  # np.less_equal(x, 3)
print(f"x >= 3: {x >= 3}")  # np.greater_equal(x, 3)
print(f"x != 3: {x != 3}")  # np.not_equal(x, 3)
print(f"x == 3: {x == 3}")  # np.equal(x, 3)

# Element-wise comparison of arrays
print(f"\nElement-wise array comparisons:")
y = np.array([0, 2, 3, 4, 6])
print(f"x = {x}")
print(f"y = {y}")
print(f"x > y: {x > y}")
print(f"x == y: {x == y}")

# Compound expressions
print(f"(2 * x) == (x ** 2): {(2 * x) == (x ** 2)}")

# 2D example
print(f"\n2D Array Comparisons:")
matrix = np.random.RandomState(0).randint(10, size=(3, 4))
print(f"Matrix:\n{matrix}")
print(f"Matrix < 6:\n{matrix < 6}")

print(f"\n=== Working with Boolean Arrays ===")

# Counting True values
print(f"Counting entries:")
mask = matrix < 6
print(f"Count of values < 6: {np.count_nonzero(mask)}")
print(f"Count using sum: {np.sum(mask)}")  # True=1, False=0
print(f"Count per row: {np.sum(mask, axis=1)}")

# Testing any/all
print(f"\nTesting conditions:")
print(f"Any values > 8? {np.any(matrix > 8)}")
print(f"Any values < 0? {np.any(matrix < 0)}")
print(f"All values < 10? {np.all(matrix < 10)}")
print(f"All values == 6? {np.all(matrix == 6)}")
print(f"All values < 8 in each row? {np.all(matrix < 8, axis=1)}")

print(f"\n=== Boolean Operators ===")

# Simulate rainfall data for practical example
np.random.seed(42)
inches = np.random.exponential(0.2, 365)  # Simulated daily rainfall

print(f"Rainfall analysis (365 days):")
print(f"Days with 0.5-1.0 inches: {np.sum((inches > 0.5) & (inches < 1))}")

# Important: parentheses required due to operator precedence!
print(f"\nParentheses are crucial:")
print("Correct:   (inches > 0.5) & (inches < 1)")
print("Incorrect: inches > 0.5 & inches < 1  # This would error!")

# De Morgan's laws demonstration
result1 = np.sum((inches > 0.5) & (inches < 1))
result2 = np.sum(~((inches <= 0.5) | (inches >= 1)))
print(f"Using AND: {result1}")
print(f"Using De Morgan's law: {result2}")
print(f"Results identical? {result1 == result2}")

# Boolean operators table
print(f"\nBoolean Operators:")
print(f"& (and):  np.bitwise_and")
print(f"| (or):   np.bitwise_or") 
print(f"^ (xor):  np.bitwise_xor")
print(f"~ (not):  np.bitwise_not")

# Practical weather analysis
print(f"\nWeather Statistics:")
print(f"Days without rain: {np.sum(inches == 0)}")
print(f"Days with rain: {np.sum(inches != 0)}")
print(f"Days with >0.5 inches: {np.sum(inches > 0.5)}")
print(f"Rainy days with <0.2 inches: {np.sum((inches > 0) & (inches < 0.2))}")

print(f"\n=== and/or vs &/| Important Distinction ===")

# Demonstrate the difference
print(f"Keywords vs Operators:")
print(f"and/or: evaluate truth of ENTIRE object")
print(f"&/|:    bitwise operations on ELEMENTS")

# Example with integers
print(f"\nWith integers:")
print(f"bool(42): {bool(42)}")
print(f"bool(0): {bool(0)}")
print(f"bool(42 and 0): {bool(42 and 0)}")  # False
print(f"bool(42 or 0): {bool(42 or 0)}")    # True

# Bitwise on integers
print(f"bin(42): {bin(42)}")
print(f"bin(59): {bin(59)}")
print(f"bin(42 & 59): {bin(42 & 59)}")
print(f"bin(42 | 59): {bin(42 | 59)}")

# With boolean arrays
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
print(f"\nWith boolean arrays:")
print(f"A: {A}")
print(f"B: {B}")
print(f"A | B: {A | B}")  # Element-wise OR

try:
    result = A or B  # This will fail
except ValueError as e:
    print(f"A or B fails: {e}")

# Correct usage for array conditions
data = np.arange(10)
print(f"\nCorrect array logic:")
print(f"(data > 4) & (data < 8): {(data > 4) & (data < 8)}")

try:
    wrong = (data > 4) and (data < 8)  # This will fail
except ValueError as e:
    print(f"(data > 4) and (data < 8) fails: {e}")

print(f"\n=== Fancy Indexing (VanderPlas Ch 2.07) ===")

# Basic fancy indexing
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(f"Array: {x}")

# Select multiple elements
indices = [3, 7, 4]
selected = x[indices]
print(f"Indices {indices}: {selected}")

# Shape of result reflects shape of index array
ind_2d = np.array([[3, 7], [4, 5]])
result_2d = x[ind_2d]
print(f"2D indices:\n{ind_2d}")
print(f"2D result:\n{result_2d}")

# 2D fancy indexing
X = np.arange(12).reshape(3, 4)
print(f"\n2D Array:\n{X}")

# Select specific elements
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
elements = X[row, col]  # (0,2), (1,1), (2,3)
print(f"Elements at coordinates: {elements}")

# Broadcasting in fancy indexing
print(f"Broadcasting with fancy indexing:")
print(f"X[row[:, np.newaxis], col]:\n{X[row[:, np.newaxis], col]}")

print(f"\n=== Combined Indexing ===")

# Mix fancy indexing with other methods
print(f"Original array:\n{X}")
print(f"X[2, [2, 0, 1]]: {X[2, [2, 0, 1]]}")  # Row 2, specific columns
print(f"X[1:, [2, 0, 1]]:\n{X[1:, [2, 0, 1]]}")  # Slice + fancy

# Fancy + boolean masking
mask = np.array([1, 0, 1, 0], dtype=bool)
print(f"X[row[:, np.newaxis], mask]:\n{X[row[:, np.newaxis], mask]}")

print(f"\n=== Modifying Values with Fancy Indexing ===")

# Modify multiple elements
x = np.arange(10)
i = np.array([2, 1, 8, 4])
print(f"Before: {x}")

x[i] = 99
print(f"After x[i] = 99: {x}")

x[i] -= 10  
print(f"After x[i] -= 10: {x}")

# Repeated indices behavior
print(f"\nRepeated indices behavior:")
x = np.zeros(10)
x[[0, 0]] = [4, 6]
print(f"x[[0, 0]] = [4, 6] → x[0] = {x[0]}")  # Last assignment wins

# Accumulating with repeated indices
i = [2, 3, 3, 4, 4, 4]
x = np.zeros(10)
x[i] += 1
print(f"x[{i}] += 1 → x = {x}")  # Each position incremented once only

# Correct way to accumulate
x = np.zeros(10)
np.add.at(x, i, 1)  # Accumulate properly
print(f"np.add.at(x, {i}, 1) → x = {x}")  # Proper accumulation

print(f"\n=== Practical Example: Random Point Selection ===")

# Generate random 2D points
mean = [0, 0]
cov = [[1, 2], [2, 5]]
points = rand.multivariate_normal(mean, cov, 100)
print(f"Generated {points.shape[0]} random points")

# Select 20 random points using fancy indexing
indices = np.random.choice(points.shape[0], 20, replace=False)
selection = points[indices]
print(f"Selected {selection.shape[0]} random points")
print(f"Selected indices: {indices[:10]}...")  # Show first 10

print(f"\n=== Data Binning Example ===")

# Efficient histogram computation
np.random.seed(42)
data = np.random.randn(100)

# Manual binning using fancy indexing
bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

# Find appropriate bin for each data point
i = np.searchsorted(bins, data)
np.add.at(counts, i, 1)

print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
print(f"Number of bins: {len(bins)}")
print(f"Counts per bin (first 10): {counts[:10]}")

# Compare with numpy histogram
np_counts, np_edges = np.histogram(data, bins)
print(f"NumPy histogram matches: {np.allclose(counts[:-1], np_counts)}")

print(f"\n=== Reshaping Operations ===")

# Basic reshaping
z = np.arange(12)
print(f"Original 1D: {z}")
mat2x6 = z.reshape(2, 6)
mat3x4 = z.reshape(3, 4)
mat4x3 = z.reshape(4, 3)
print(f"Reshaped 2×6:\n{mat2x6}")
print(f"Reshaped 3×4:\n{mat3x4}")
print(f"Reshaped 4×3:\n{mat4x3}")

# Automatic dimension calculation
auto_reshape = z.reshape(3, -1)  # -1 means "figure it out"
print(f"Auto reshape (3, -1):\n{auto_reshape}")

# Flatten vs ravel
original = mat3x4.copy()
flat_copy = original.flatten()  # Always returns a copy
flat_view = original.ravel()    # Returns view if possible

print(f"Original:\n{original}")
flat_copy[0] = 999
flat_view[1] = 888
print(f"After modifying flattened arrays:")
print(f"Original after flatten modify:\n{original}")  # unchanged
print(f"Original after ravel modify:\n{original}")    # changed at position 1

print(f"\n=== Array Joining ===")

# Stacking operations
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(f"Array 1:\n{arr1}")
print(f"Array 2:\n{arr2}")

# Different stacking methods
vstack_result = np.vstack([arr1, arr2])  # Vertical stack
hstack_result = np.hstack([arr1, arr2])  # Horizontal stack
dstack_result = np.dstack([arr1, arr2])  # Depth stack (along 3rd dimension)

print(f"Vertical stack (vstack):\n{vstack_result}")
print(f"Horizontal stack (hstack):\n{hstack_result}")
print(f"Depth stack (dstack) shape: {dstack_result.shape}")

# Concatenation with axis specification
concat_axis0 = np.concatenate([arr1, arr2], axis=0)  # Same as vstack
concat_axis1 = np.concatenate([arr1, arr2], axis=1)  # Same as hstack
print(f"Concatenate axis=0:\n{concat_axis0}")
print(f"Concatenate axis=1:\n{concat_axis1}")

# Stack creates new dimension
stack_axis0 = np.stack([arr1, arr2], axis=0)  # Shape: (2, 2, 2)
stack_axis1 = np.stack([arr1, arr2], axis=1)  # Shape: (2, 2, 2)
stack_axis2 = np.stack([arr1, arr2], axis=2)  # Shape: (2, 2, 2)
print(f"Stack axis=0 shape: {stack_axis0.shape}")
print(f"Stack axis=1 shape: {stack_axis1.shape}")
print(f"Stack axis=2 shape: {stack_axis2.shape}")

print(f"\n=== Array Splitting ===")

# Splitting arrays
data_1d = np.arange(12)
print(f"1D data: {data_1d}")

# Equal splits
equal_splits = np.split(data_1d, 4)  # Split into 4 equal parts
print(f"Split into 4 equal parts: {[arr.tolist() for arr in equal_splits]}")

# Split at specific indices
index_splits = np.split(data_1d, [3, 7])  # Split at indices 3 and 7
print(f"Split at indices [3,7]: {[arr.tolist() for arr in index_splits]}")

# 2D splitting
data_2d = np.arange(24).reshape(4, 6)
print(f"2D data shape: {data_2d.shape}")

# Split along different axes
vsplit_result = np.vsplit(data_2d, 2)  # Split vertically (along rows)
hsplit_result = np.hsplit(data_2d, 3)  # Split horizontally (along columns)

print(f"Vertical split into 2 parts: {[arr.shape for arr in vsplit_result]}")
print(f"Horizontal split into 3 parts: {[arr.shape for arr in hsplit_result]}")

# Array splitting with unequal parts
unequal_splits = np.array_split(data_1d, 5)  # 5 parts (some may be different sizes)
print(f"Unequal splits: {[arr.tolist() for arr in unequal_splits]}")

print(f"\n=== Advanced Reshaping ===")

# Transpose and swapping axes
matrix_3d = np.arange(24).reshape(2, 3, 4)
print(f"3D array shape: {matrix_3d.shape}")

# Different transpose operations
transposed = matrix_3d.transpose()  # Reverse all axes
transposed_custom = matrix_3d.transpose(2, 0, 1)  # Specify axis order
swapped = matrix_3d.swapaxes(0, 2)  # Swap axes 0 and 2

print(f"Original shape: {matrix_3d.shape}")
print(f"Transposed shape: {transposed.shape}")
print(f"Custom transpose (2,0,1): {transposed_custom.shape}")
print(f"Swapped axes 0&2: {swapped.shape}")

# Adding and removing dimensions
print(f"\n=== Dimension Manipulation ===")
vector = np.array([1, 2, 3, 4])
print(f"Original vector shape: {vector.shape}")

# Add dimensions
col_vector = vector[:, np.newaxis]  # Add column dimension
row_vector = vector[np.newaxis, :]  # Add row dimension
matrix_form = vector[:, np.newaxis, np.newaxis]  # Add multiple dimensions

print(f"Column vector shape: {col_vector.shape}")
print(f"Row vector shape: {row_vector.shape}")
print(f"Matrix form shape: {matrix_form.shape}")

# Remove dimensions (squeeze)
squeezed = np.squeeze(matrix_form)  # Remove single dimensions
print(f"After squeeze: {squeezed.shape}")
```

**Exercise**  
1. Create a 3×3 array with values 0–8, use fancy indexing to pick elements at positions [(0,2), (1,1), (2,0)].
2. Split the 3×3 array into three 1×3 subarrays.
3. Create two 2×3 arrays, stack them vertically, then split the result back into the original arrays.
4. Use reshaping to convert a 1D array of 24 elements into a 3D array of shape (2,3,4).

**Answer**  
```python
# 1. Fancy indexing on 3×3 array
x = np.arange(9).reshape(3, 3)
print("Original 3×3 array:")
print(x)

# Method 1: Using coordinate arrays
rows = [0, 1, 2]
cols = [2, 1, 0]
selected = x[rows, cols]
print(f"Selected elements at (0,2), (1,1), (2,0): {selected}")  # [2, 4, 6]

# Method 2: Using list of tuples (alternative approach)
positions = [(0, 2), (1, 1), (2, 0)]
selected_alt = [x[r, c] for r, c in positions]
print(f"Alternative selection: {selected_alt}")

# 2. Split into 1×3 subarrays
splits = np.split(x, 3, axis=0)  # Split along rows
print(f"Split shapes: {[s.shape for s in splits]}")  # [(1,3), (1,3), (1,3)]
print("Split arrays:")
for i, split in enumerate(splits):
    print(f"  Split {i}: {split}")

# 3. Stack and split operations
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
print(f"\nOriginal arrays:")
print(f"Array 1:\n{arr1}")
print(f"Array 2:\n{arr2}")

# Stack vertically
stacked = np.vstack([arr1, arr2])
print(f"Stacked shape: {stacked.shape}")
print(f"Stacked array:\n{stacked}")

# Split back to original
split_back = np.split(stacked, 2, axis=0)
reconstructed1, reconstructed2 = split_back
print(f"Reconstructed arrays:")
print(f"Reconstructed 1:\n{reconstructed1}")
print(f"Reconstructed 2:\n{reconstructed2}")

# Verify they match
print(f"Array 1 matches: {np.array_equal(arr1, reconstructed1)}")
print(f"Array 2 matches: {np.array_equal(arr2, reconstructed2)}")

# 4. 1D to 3D reshaping
data_1d = np.arange(24)
print(f"\n1D array: {data_1d}")
print(f"1D shape: {data_1d.shape}")

# Reshape to 3D
data_3d = data_1d.reshape(2, 3, 4)
print(f"3D shape: {data_3d.shape}")
print(f"3D array:\n{data_3d}")

# Verify total elements preserved
print(f"Original size: {data_1d.size}")
print(f"3D size: {data_3d.size}")
print(f"Sizes match: {data_1d.size == data_3d.size}")

# Alternative reshape methods
alt_3d_1 = data_1d.reshape(-1, 3, 4)  # Auto-calculate first dimension
alt_3d_2 = data_1d.reshape(2, -1, 4)  # Auto-calculate second dimension
print(f"Alternative shapes: {alt_3d_1.shape}, {alt_3d_2.shape}")
```  

### Key Points
- Advanced indexing allows complex data selection and manipulation patterns
- Reshaping preserves total number of elements but changes array dimensions
- Use `-1` in reshape to automatically calculate one dimension
- Stacking functions (`vstack`, `hstack`, `dstack`) combine arrays along different axes
- Splitting functions reverse stacking operations and are useful for data partitioning

---

## 6. Performance tuning and mini‑project (16:00–16:50)

### Questions
- How can I optimize NumPy code for better performance?
- What memory layout considerations affect array operations?
- How do I apply NumPy skills to solve real-world problems?

### Objectives
After completing this episode, learners will be able to:
- Apply performance optimization techniques for NumPy operations
- Understand memory layout and its impact on performance
- Complete a mini-project using NumPy skills learned throughout the workshop

**Hands‑On Code**  
```python
import numpy as np

# Performance comparison
print("=== Performance Optimization ===")

# Memory layout considerations
print("Memory layout examples...")
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')  # C-style (row-major)
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')  # Fortran-style (column-major)

print(f"C-style flags: {arr_c.flags}")
print(f"F-style flags: {arr_f.flags}")

# Mini-project: Data analysis pipeline
print(f"\n=== Mini-Project: Weather Data Analysis ===")
# This would reference the weather notebook exercise
print("Apply your NumPy skills to analyze weather data!")
```

### Key Points - Performance
- Memory layout (C vs Fortran order) affects performance
- Vectorized operations are much faster than Python loops
- Use views instead of copies when possible
- Apply NumPy skills to real-world data analysis problems

---

## 7. Wrap-up (16:50–17:00)

### Summary
- NumPy arrays: creation, indexing, views, and memory
- Universal functions, aggregations, and performance
- Advanced indexing, reshaping, and real-world applications

### Further Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [SciPy Lecture Notes](https://scipy-lectures.org/)
- [Awesome NumPy](https://github.com/hamelsmu/awesome-numpy)

### Feedback & Q&A
- What did you find most useful?
- Any remaining questions or topics?
- Suggestions for future workshops?

---