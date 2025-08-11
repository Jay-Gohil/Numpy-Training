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
```

**Exercise**  
Run the code above. Confirm that you’re running Python 3.6+ and NumPy 1.15+ (or the latest installed), and that the test array prints correctly.  

**Answer**  
```text
Python version: 3.13.1
NumPy version: 2.3.1
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

### Why is NumPy Faster?

Let's compare doing the same task with Python lists vs NumPy arrays:

```python
import numpy as np

# Task: Double all numbers from 0 to 999
print("=== Doubling 1000 numbers ===")

# Method 1: Python list with loop
python_numbers = list(range(1000))
python_doubled = []
for num in python_numbers:
    python_doubled.append(num * 2)

print(f"Python approach: Created list of {len(python_doubled)} numbers")
print(f"First 5 results: {python_doubled[:5]}")

# Method 2: NumPy array (vectorized)
numpy_numbers = np.arange(1000)
numpy_doubled = numpy_numbers * 2  # This multiplies ALL numbers at once!

print(f"NumPy approach: Created array of {len(numpy_doubled)} numbers")
print(f"First 5 results: {numpy_doubled[:5]}")
print(f"Results are the same: {python_doubled[:5] == numpy_doubled[:5].tolist()}")
```

### Key Differences

```python
# 1. Code simplicity
print("\n=== Code Comparison ===")
print("Python: Need a loop to process each number")
print("NumPy:  One operation processes ALL numbers")

# 2. Speed demonstration (simple version)
import timeit

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

# 3. Memory efficiency
# np. arange: Return evenly spaced values within a given interval
numbers = np.arange(1000)
print(f"\n=== Memory Usage ===")
print(f"NumPy array uses: {numbers.nbytes} bytes")
print(f"That's only {numbers.nbytes/1000:.1f} bytes per number!")
```

### When Should You Use NumPy?

```python
# NumPy is great for:
print("\n=== When to Use NumPy ===")

# Large amounts of numerical data
temperatures = np.array([20.5, 22.1, 19.8, 25.0, 23.2])
print(f"Daily temperatures: {temperatures}")
print(f"Average temperature: {temperatures.mean():.1f}°C")
print(f"Hottest day: {temperatures.max():.1f}°C")

# Mathematical operations on many numbers
scores = np.array([85, 92, 78, 96, 88])
print(f"\nTest scores: {scores}")
print(f"Add 5 bonus points to all: {scores + 5}")
print(f"Convert to percentages: {scores}%")

# Working with grids/tables of data
sales_data = np.array([[100, 150, 200],    # Store 1: Jan, Feb, Mar
                       [120, 130, 180],    # Store 2: Jan, Feb, Mar  
                       [90,  140, 220]])   # Store 3: Jan, Feb, Mar

print(f"\nSales data shape: {sales_data.shape} (3 stores × 3 months)")
print(f"Total sales per store: {sales_data.sum(axis=1)}")
print(f"Total sales per month: {sales_data.sum(axis=0)}")
```

### Exercise
1. Create a NumPy array with numbers 1 to 10, then multiply all numbers by 3
2. Compare how you would calculate the average of 5 test scores using Python lists vs NumPy
3. Try using NumPy to add 10 to every element in an array

#### Answer

```python
# 1. Create array and multiply by 3
numbers = np.arange(1, 11)  # Creates [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tripled = numbers * 3
print(f"Original: {numbers}")
print(f"Tripled:  {tripled}")

# 2. Average calculation comparison
test_scores = [85, 92, 78, 96, 88]

# Python way
python_average = sum(test_scores) / len(test_scores)
print(f"Python average: {python_average:.1f}")

# NumPy way
np_scores = np.array(test_scores)
numpy_average = np_scores.mean()
print(f"NumPy average: {numpy_average:.1f}")

# 3. Add 10 to every element
original = np.array([1, 5, 10, 15, 20])
plus_ten = original + 10
print(f"Original: {original}")
print(f"Plus 10:  {plus_ten}")
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
print(f"\nnp.linspace() - Creates evenly spaced numbers")
print("Key difference: linspace specifies NUMBER of points, arange specifies STEP size")

# Basic linspace usage
linspace_basic = np.linspace(0, 1, 5)     # 5 points from 0 to 1 (inclusive)
print(f"  linspace(0, 1, 5): {linspace_basic}")
print(f"    Creates exactly 5 points from 0 to 1")

# More linspace examples
linspace_10 = np.linspace(0, 10, 11)      # 11 points from 0 to 10
linspace_pi = np.linspace(0, np.pi, 4)    # 4 points from 0 to π
linspace_exclusive = np.linspace(0, 1, 5, endpoint=False)  # Exclude endpoint

print(f"  linspace(0, 10, 11): {linspace_10}")
print(f"  linspace(0, π, 4): {linspace_pi}")
print(f"  linspace(0, 1, 5, endpoint=False): {linspace_exclusive}")

# Practical comparison: arange vs linspace
print(f"\n=== arange vs linspace comparison ===")
print("Creating points from 0 to 2:")
print(f"arange(0, 2.5, 0.5): {np.arange(0, 2.5, 0.5)}")     # Step-based
print(f"linspace(0, 2, 5): {np.linspace(0, 2, 5)}")         # Count-based

print(f"\nWhen to use which:")
print(f"  arange: When you know the STEP size you want")
print(f"  linspace: When you know the NUMBER of points you want")

# Getting both values and step size
values, step = np.linspace(0, 1, 5, retstep=True)
print(f"\nWith retstep=True:")
print(f"  Values: {values}")
print(f"  Step size: {step}")
```

### Special Arrays

```python
# Identity matrices
identity_3x3 = np.eye(3)                  # 3x3 identity matrix
print("np.eye() - Creates identity matrices")
print(f"  3x3 identity:\n{identity_3x3}")

# Random arrays
rng = np.random.default_rng(42)  # Create random number generator with seed
random_uniform = rng.random((2, 3))         # Uniform [0, 1)
random_normal = rng.normal(0, 1, (2, 3))    # Normal distribution (mean=0, std=1)

print(f"\nRandom arrays:")
print(f"  Uniform random:\n{random_uniform}")
print(f"  Normal distribution:\n{random_normal}")

# More random array examples
random_integers = rng.integers(1, 10, size=(2, 4))  # Random integers 1-9
random_choice = rng.choice(['A', 'B', 'C'], size=5)  # Random choices
print(f"  Random integers (1-9):\n{random_integers}")
print(f"  Random choices: {random_choice}")
```

### Array Inspection

```python
# Create sample array for inspection using modern random API
rng = np.random.default_rng(42)
sample_array = rng.normal(0, 1, (3, 4)).astype(np.float32)

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
# Understanding NumPy data types and their characteristics
print("=== NumPy Data Types Overview ===")

# Automatic type inference
int_array = np.array([1, 2, 3, 4])
float_array = np.array([1.0, 2.5, 3.7])
mixed_array = np.array([1, 2.5, 3])  # Promotes to float

print(f"Automatic type inference:")
print(f"Integer array: {int_array.dtype}")
print(f"Float array: {float_array.dtype}")
print(f"Mixed array: {mixed_array.dtype} (promoted to float)")

# Integer types and their ranges
print(f"\n=== Integer Data Types ===")
integer_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]

for dtype in integer_types:
    info = np.iinfo(dtype)
    arr = np.array([100], dtype=dtype)
    print(f"{dtype.__name__:<8}: range [{info.min:>20}, {info.max:>20}], size: {arr.itemsize} bytes")

# Float types and their precision
print(f"\n=== Floating Point Data Types ===")
float_types = [np.float16, np.float32, np.float64]

for dtype in float_types:
    info = np.finfo(dtype)
    arr = np.array([3.14159], dtype=dtype)
    print(f"{dtype.__name__:<9}: precision: ~{info.precision} digits, range: ~1e{int(info.maxexp*0.3)}, size: {arr.itemsize} bytes")
    print(f"            Value: {arr[0]}")

# Boolean and complex types
print(f"\n=== Other Data Types ===")
bool_array = np.array([True, False, True], dtype=np.bool_)
complex_array = np.array([1+2j, 3+4j], dtype=np.complex128)

print(f"Boolean: {bool_array.dtype}, size: {bool_array.itemsize} byte per element")
print(f"Complex: {complex_array.dtype}, size: {complex_array.itemsize} bytes per element")

# Data type drawbacks and limitations
print(f"\n=== Data Type Drawbacks and Limitations ===")

# 1. Integer overflow
print("1. Integer Overflow:")
small_int = np.array([127], dtype=np.int8)
print(f"   int8 value: {small_int[0]}")
try:
    overflowed = small_int + np.array([1], dtype=np.int8)
    print(f"   127 + 1 = {overflowed[0]} (overflow! wraps to negative)")
except:
    print("   Overflow detected")

# 2. Precision loss in floats
print(f"\n2. Floating Point Precision Loss:")
precise_value = np.pi
float32_value = np.array([precise_value], dtype=np.float32)
float64_value = np.array([precise_value], dtype=np.float64)

print(f"   Original π: {precise_value}")
print(f"   float32 π:  {float32_value[0]}")
print(f"   float64 π:  {float64_value[0]}")
print(f"   Precision lost in float32: {abs(precise_value - float32_value[0]):.2e}")

# 3. Memory vs precision trade-offs
print(f"\n3. Memory vs Precision Trade-offs:")
large_data = np.ones(1_000_000)
large_float32 = large_data.astype(np.float32)
large_float64 = large_data.astype(np.float64)

print(f"   1M elements as float32: {large_float32.nbytes / 1024**2:.1f} MB")
print(f"   1M elements as float64: {large_float64.nbytes / 1024**2:.1f} MB")
print(f"   Memory savings with float32: {(1 - large_float32.nbytes/large_float64.nbytes)*100:.0f}%")

# 4. Type casting dangers
print(f"\n4. Dangerous Type Casting:")
float_data = np.array([3.7, 4.9, 5.1])
int_cast = float_data.astype(np.int32)  # Truncation, not rounding!
print(f"   Original floats: {float_data}")
print(f"   Cast to int32:   {int_cast} (truncated, not rounded!)")

# 5. Platform dependencies
print(f"\n5. Platform Dependencies:")
print(f"   Default int: {np.array([1]).dtype} (platform dependent)")
print(f"   Default float: {np.array([1.0]).dtype} (usually float64)")

# Best practices
print(f"\n=== Best Practices ===")
```

### Best Practices for NumPy Data Types

**Do's:**
- **Be explicit about data types** when precision matters for calculations
- **Use the smallest suitable type** to save memory for large arrays
- **Be aware of overflow** in integer operations, especially with small integer types
- **Use float64** for calculations requiring high precision (scientific computing)
- **Consider float32** for large datasets where precision requirements allow
- **Check type ranges** using `np.iinfo()` and `np.finfo()` when working with edge values
- **Plan for memory usage** in advance for large-scale data processing

**Don'ts:**
- **Avoid implicit type casting** that might lose data or precision
- **Don't assume platform-independent behavior** with default types
- **Don't ignore overflow warnings** - they indicate potential data corruption
- **Avoid mixing types unnecessarily** - it can lead to unexpected promotions
- **Don't use oversized types** when smaller ones suffice (memory waste)

**Type Selection Guidelines:**
- **uint8**: Image data, sensor readings (0-255 range)
- **int32/int64**: General integer calculations, indices
- **float32**: Graphics, machine learning (when precision allows)
- **float64**: Scientific computing, financial calculations
- **bool**: Logical operations, masks
- **complex**: Signal processing, mathematical computations

```python

### Exercise
1. Create a 3×4 array filled with the value 7 using np.full, then report min, max, and mean
2. Create a 5×5 identity matrix and modify it to have 2s on the diagonal  
3. Use `np.linspace()` to create 50 evenly spaced points between 0 and 2π, then compute sin of each point
4. Compare `np.arange(0, 1, 0.1)` vs `np.linspace(0, 1, 11)` - what's the difference?

#### Answer

```python
# 1. Array filled with 7 using np.full()
sevens_array = np.full((3, 4), 7)
print(f"Array filled with 7s:\n{sevens_array}")
print(f"Min: {sevens_array.min()}, Max: {sevens_array.max()}, Mean: {sevens_array.mean()}")

# 2. Modified identity matrix
identity = np.eye(5)
print(f"Original identity matrix:\n{identity}")

# In-place multiplication
identity *= 2  # Scale diagonal by 2
print(f"2x Identity matrix:\n{identity}")

# 3. Linspace for trigonometric functions
x = np.linspace(0, 2*np.pi, 50)  # 50 points from 0 to 2π
y = np.sin(x)
print(f"X values (first 5): {x[:5]}")
print(f"Sin values (first 5): {y[:5]}")
print(f"Created {len(x)} points for smooth sin curve")

# 4. arange vs linspace comparison
arange_result = np.arange(0, 1, 0.1)
linspace_result = np.linspace(0, 1, 11)

print(f"\narange(0, 1, 0.1): {arange_result}")
print(f"Length: {len(arange_result)}")
print(f"Last value: {arange_result[-1]:.1f}")

print(f"\nlinspace(0, 1, 11): {linspace_result}")
print(f"Length: {len(linspace_result)}")
print(f"Last value: {linspace_result[-1]:.1f}")

print(f"\nKey differences:")
print(f"- arange: step-based, may not include endpoint (0.9 vs 1.0)")
print(f"- linspace: count-based, always includes endpoint")
print(f"- arange gives 10 points, linspace gives exactly 11 points")
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
rng = np.random.default_rng(42)  # Modern random number generator
big_array = rng.random(1000000)
print(f"Performance comparison on 1M elements:")

# Time comparison (conceptual - actual timing would vary)
# Python sum: ~100ms, NumPy sum: ~500μs (200x faster)
print(f"Sum result: {np.sum(big_array):.4f}")
print(f"NumPy sum is ~200x faster than Python's built-in sum()")

# Comprehensive aggregation table (following VanderPlas Table 2-3)
print(f"\nComprehensive aggregations on random data (n=1000):")
data = rng.standard_normal(1000)  # Modern way to generate normal random data

# Table format following VanderPlas style
aggregations = [
    ('np.sum', np.sum(data), 'np.nansum', 'Sum of elements'),
    ('np.prod', np.prod(data[:10]), 'np.nanprod', 'Product of elements'),  # Use subset for product
    ('np.mean', np.mean(data), 'np.nanmean', 'Mean of elements'),
    ('np.std', np.std(data), 'np.nanstd', 'Standard deviation'),
    ('np.var', np.var(data), 'np.nanvar', 'Variance'),
    ('np.min', np.min(data), 'np.nanmin', 'Minimum value'),
    ('np.max', np.max(data), 'np.nanmax', 'Maximum value'),
    ('np.argmin', np.argmin(data), 'np.nanargmin', 'Index of minimum'),
    ('np.argmax', np.argmax(data), 'np.nanargmax', 'Index of maximum'),
    ('np.median', np.median(data), 'np.nanmedian', 'Median value'),
    ('np.percentile', np.percentile(data, 25), 'np.nanpercentile', '25th percentile'),
    ('np.any', np.any(data > 0), 'N/A', 'Any elements True'),
    ('np.all', np.all(data > -5), 'N/A', 'All elements True')
]

print(f"{'Function':<15} {'Result':<12} {'NaN-safe':<15} {'Description'}")
print("-" * 70)
for func, result, nan_safe, desc in aggregations:
    if isinstance(result, (int, np.integer)):
        result_str = f"{result}"
    elif isinstance(result, bool):
        result_str = f"{result}"
    else:
        result_str = f"{result:.4f}"
    print(f"{func:<15} {result_str:<12} {nan_safe:<15} {desc}")

print(f"\nNote: NaN-safe versions ignore NaN values in calculations")
print(f"      Use them when your data might contain missing values")

# Multi-dimensional aggregations with axis examples
print(f"\n=== Multi-dimensional Aggregations ===")
matrix = rng.random((3, 4))  # Using the same rng instance
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

# Practical example: US Presidents Heights (following VanderPlas Ch 2.04)
print(f"\n=== Practical Example: US President Heights Analysis ===")

# Loading data from CSV files (real-world skill)
print("=== Loading Data from CSV ===")
print("Loading president heights using NumPy:")
print("  heights = np.loadtxt('president_heights.csv', delimiter=',', skiprows=1, usecols=2)")

# Load the actual president heights data from CSV
president_heights = np.loadtxt('president_heights.csv', delimiter=',', skiprows=1, usecols=2)

print(f"Loaded {len(president_heights)} president heights from CSV file")
print(f"Data preview: {president_heights[:5]}... (first 5 values)")
print(f"Data source: president_heights.csv")

# Basic aggregations following VanderPlas examples
print(f"\n=== Basic Statistical Aggregations ===")
print(f"Mean height: {president_heights.mean():.1f} cm")
print(f"Standard deviation: {president_heights.std():.1f} cm")
print(f"Minimum height: {president_heights.min()} cm ({np.argmin(president_heights) + 1}th president)")
print(f"Maximum height: {president_heights.max()} cm ({np.argmax(president_heights) + 1}th president)")

# Quantile aggregations
print(f"\n=== Quantile Analysis ===")
print(f"25th percentile: {np.percentile(president_heights, 25):.1f} cm")
print(f"Median (50th percentile): {np.median(president_heights):.1f} cm")  
print(f"75th percentile: {np.percentile(president_heights, 75):.1f} cm")

# Alternative quantile method
print(f"Median (alternative): {np.quantile(president_heights, 0.5):.1f} cm")
print(f"Interquartile range: {np.percentile(president_heights, 75) - np.percentile(president_heights, 25):.1f} cm")

# Boolean aggregations following VanderPlas pattern
print(f"\n=== Boolean Aggregations (VanderPlas Pattern) ===")
print(f"How many presidents are taller than the mean?")
tall_presidents = president_heights > president_heights.mean()
print(f"  Answer: {np.sum(tall_presidents)} out of {len(president_heights)}")
print(f"  Percentage: {100 * np.mean(tall_presidents):.1f}%")

print(f"\nHow many presidents are shorter than 170cm?")
short_presidents = president_heights < 170
print(f"  Answer: {np.sum(short_presidents)} presidents")

print(f"\nHow many presidents are between 175cm and 185cm?")
medium_presidents = (president_heights >= 175) & (president_heights <= 185)
print(f"  Answer: {np.sum(medium_presidents)} presidents")

# VanderPlas-style aggregation questions
print(f"\n=== VanderPlas-Style Analysis Questions ===")
print(f"Are there any presidents taller than 190cm? {np.any(president_heights > 190)}")
print(f"Are all presidents taller than 160cm? {np.all(president_heights > 160)}")
print(f"Are all presidents shorter than 200cm? {np.all(president_heights < 200)}")

# Historical insights using boolean indexing
print(f"\n=== Historical Insights ===")
very_tall = president_heights >= 190
print(f"Very tall presidents (≥190cm): {np.sum(very_tall)} total")
if np.any(very_tall):
    tall_indices = np.where(very_tall)[0]
    print(f"  President numbers: {[i+1 for i in tall_indices]}")  # 1-indexed

very_short = president_heights <= 170
print(f"Very short presidents (≤170cm): {np.sum(very_short)} total")
if np.any(very_short):
    short_indices = np.where(very_short)[0]
    print(f"  President numbers: {[i+1 for i in short_indices]}")  # 1-indexed

# Working with missing data (NaN-safe functions) - VanderPlas Section 2.04
print(f"\n=== Working with Missing Data (NaN-safe aggregations) ===")
# Simulate some missing data for demonstration
heights_with_missing = president_heights.astype(float)
heights_with_missing[[4, 12, 22]] = np.nan  # James Monroe, Fillmore, Cleveland

print(f"Original data points: {len(president_heights)}")
print(f"Data with missing values: {np.sum(np.isnan(heights_with_missing))} NaN values")
print(f"Valid data points: {np.sum(~np.isnan(heights_with_missing))}")

print(f"\nComparison of regular vs NaN-safe functions:")
print(f"Regular mean (with NaN): {np.mean(heights_with_missing)}")  # Returns NaN
print(f"NaN-safe mean: {np.nanmean(heights_with_missing):.1f} cm")
print(f"Regular std (with NaN): {np.std(heights_with_missing)}")   # Returns NaN
print(f"NaN-safe std: {np.nanstd(heights_with_missing):.1f} cm")
print(f"NaN-safe median: {np.nanmedian(heights_with_missing):.1f} cm")
print(f"NaN-safe min: {np.nanmin(heights_with_missing):.0f} cm")
print(f"NaN-safe max: {np.nanmax(heights_with_missing):.0f} cm")

# Cumulative operations
arr = np.array([1, 2, 3, 4, 5])
print(f"\nCumulative operations on {arr}:")
print(f"Cumulative sum: {np.cumsum(arr)}")
print(f"Cumulative product: {np.cumprod(arr)}")
```

### Understanding Cumulative Operations

Cumulative operations compute running totals or products as they move through an array. Unlike regular aggregations that reduce arrays to single values, cumulative operations preserve the array size while showing progressive calculations.

**Cumulative Sum (`cumsum`):**
- Takes each element and adds it to the sum of all previous elements
- Example: `[1, 2, 3, 4, 5]` becomes `[1, 3, 6, 10, 15]`
- Useful for: running totals, integration approximation, progress tracking

**Cumulative Product (`cumprod`):**
- Multiplies each element by the product of all previous elements  
- Example: `[1, 2, 3, 4, 5]` becomes `[1, 2, 6, 24, 120]`
- Useful for: compound interest, probability chains, factorial calculations

### Array API Standard vs Legacy Functions

NumPy is transitioning to the **Array API Standard** - a cross-library specification that ensures consistent behavior across different array libraries (NumPy, CuPy, JAX, etc.). This standardization makes code more portable and predictable.

**Key Differences:**
- **Legacy**: `np.cumsum()` - established function, universally available
- **Array API**: `np.cumulative_sum()` - standardized name, explicit and clear
- **Compatibility**: Array API functions require NumPy 1.22+ but provide better interoperability

**Why the Change?**
The Array API standard aims to create consistency across the scientific Python ecosystem. While legacy functions remain supported, new projects may benefit from using Array API functions for future compatibility.

```python

# Array API vs Legacy API comparison
print(f"\n=== Array API vs Legacy API ===")
print("NumPy provides two ways to compute cumulative operations:")

# Legacy API (traditional NumPy)
legacy_cumsum = np.cumsum(arr)
print(f"Legacy API - np.cumsum(): {legacy_cumsum}")

# Array API Standard (newer, more explicit)
try:
    array_api_cumsum = np.cumulative_sum(arr)
    print(f"Array API - np.cumulative_sum(): {array_api_cumsum}")
    print(f"Results are identical: {np.array_equal(legacy_cumsum, array_api_cumsum)}")
except AttributeError:
    print("Array API - np.cumulative_sum(): Not available in this NumPy version")
    print("(Requires NumPy 1.22+ for Array API compliance)")
```

### Multi-dimensional Cumulative Operations

When working with multi-dimensional arrays, cumulative operations can be applied along specific axes, allowing for sophisticated data analysis patterns.

**Axis Parameter:**
- `axis=0`: Operates along rows (down columns)
- `axis=1`: Operates along columns (across rows)  
- No axis: Flattens array first, then operates

**Common Use Cases:**
- **Financial Data**: Running totals of daily transactions by month
- **Scientific Data**: Accumulated measurements over time or space
- **Image Processing**: Progressive filters or cumulative transformations

```python

# Multi-dimensional cumulative operations
print(f"\n=== Multi-dimensional cumulative operations ===")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original matrix:\n{matrix}")

# Cumulative sum along different axes
cumsum_axis0 = np.cumsum(matrix, axis=0)  # Along rows
cumsum_axis1 = np.cumsum(matrix, axis=1)  # Along columns
cumsum_flat = np.cumsum(matrix)           # Flattened

print(f"Cumsum along axis=0 (rows):\n{cumsum_axis0}")
print(f"Cumsum along axis=1 (cols):\n{cumsum_axis1}")
print(f"Cumsum flattened: {cumsum_flat}")

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
4. Load the president heights data and answer: "What percentage of presidents are above average height?"

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

# 4. President heights analysis (VanderPlas style)
heights = np.loadtxt('president_heights.csv', delimiter=',', skiprows=1, usecols=2)

mean_height = heights.mean()
above_average = heights > mean_height
percentage_above = 100 * np.mean(above_average)

print(f"\nPresident Heights Analysis:")
print(f"Mean height: {mean_height:.1f} cm")
print(f"Presidents above average: {np.sum(above_average)} out of {len(heights)}")
print(f"Percentage above average: {percentage_above:.1f}%")

# Bonus: Tallest and shortest presidents
print(f"Tallest president: {heights.max()} cm (#{np.argmax(heights) + 1})")
print(f"Shortest president: {heights.min()} cm (#{np.argmin(heights) + 1})")
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
rng = np.random.default_rng(42)  # Create new RNG for consistency
X = rng.random((10, 3))  # 10 observations, 3 features
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
points = rng.random((5, 2))  # 5 points in 2D
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
data = rng.standard_normal((6, 4))  # 6 rows, 4 columns
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

## 5. Advanced indexing and reshaping (15:00–16:00)

### Questions
- How can I select specific data from arrays using conditions?
- When would I need to change the shape of my arrays?
- How do I combine data from multiple arrays?

### Objectives
After completing this episode, learners will be able to:
- Use boolean indexing to filter data based on conditions
- Select specific elements using fancy indexing
- Reshape arrays for different data analysis needs
- Combine and split arrays for data processing workflows

**Hands‑On Code**  

### Part A: Boolean Indexing - Filtering Your Data

```python
import numpy as np

# Real-world example: Student grades
print("=== Student Grade Analysis ===")
student_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eva', 'Frank']
grades = np.array([85, 92, 78, 96, 88, 74])

print(f"Students: {student_names}")
print(f"Grades: {grades}")

# Find students who passed (grade >= 80)
passing_mask = grades >= 80
print(f"\nPassing grades mask: {passing_mask}")
print(f"Passing grades: {grades[passing_mask]}")

# Which students passed? (using same mask on names)
passing_students = np.array(student_names)[passing_mask]
print(f"Students who passed: {list(passing_students)}")

# Multiple conditions: Students with grades between 80-90
good_grades = (grades >= 80) & (grades <= 90)
print(f"Students with 80-90: {grades[good_grades]}")

# Modify grades: Add 5 points to failing students
print(f"\nOriginal grades: {grades}")
grades[grades < 80] += 5  # Boolean indexing for modification
print(f"After bonus points: {grades}")
```

### Part B: Fancy Indexing - Selecting Specific Elements

```python
print("\n=== Fancy Indexing Examples ===")

# Example: Monthly sales data for different stores
monthly_sales = np.array([
    [120, 135, 142, 155],  # Store A: Jan, Feb, Mar, Apr
    [98,  108, 125, 140],  # Store B: Jan, Feb, Mar, Apr  
    [145, 152, 138, 160]   # Store C: Jan, Feb, Mar, Apr
])

print("Monthly sales (rows=stores, cols=months):")
print(monthly_sales)

# Select specific stores (rows 0 and 2)
selected_stores = monthly_sales[[0, 2]]
print(f"\nStores A and C only:\n{selected_stores}")

# Select specific months (columns 1 and 3 = Feb and Apr)
selected_months = monthly_sales[:, [1, 3]]
print(f"\nFeb and Apr data:\n{selected_months}")

# Select specific store-month combinations
store_indices = [0, 1, 2]  # Store A, B, C
month_indices = [0, 2, 3]  # Jan, Mar, Apr
specific_data = monthly_sales[store_indices, month_indices]
print(f"\nStore A-Jan, Store B-Mar, Store C-Apr: {specific_data}")
```

### Part C: Reshaping - Changing Array Dimensions

```python
print("\n=== Reshaping for Different Analysis ===")

# Example: Daily temperature data for a week
daily_temps = np.array([22, 24, 26, 25, 23, 21, 20, 
                       23, 25, 27, 26, 24, 22, 21])
print(f"14 days of temperature data: {daily_temps}")

# Reshape into 2 weeks × 7 days for weekly analysis
weekly_temps = daily_temps.reshape(2, 7)
print(f"\nReshaped into weeks:\n{weekly_temps}")
print(f"Week 1 average: {weekly_temps[0].mean():.1f}°C")
print(f"Week 2 average: {weekly_temps[1].mean():.1f}°C")

# Reshape into 7 days × 2 weeks for daily patterns
daily_pattern = daily_temps.reshape(7, 2)
print(f"\nDaily patterns across weeks:\n{daily_pattern}")
print(f"Monday temperatures: {daily_pattern[0]}")
print(f"Sunday temperatures: {daily_pattern[6]}")

# Automatic dimension calculation using -1
auto_reshape = daily_temps.reshape(-1, 2)  # NumPy calculates first dimension
print(f"\nAuto reshape (-1, 2): shape {auto_reshape.shape}")
```

### Part D: Combining Arrays - Stacking Data

```python
print("\n=== Combining Data from Different Sources ===")

# Example: Combining test scores from different classes
class_a_scores = np.array([85, 92, 78, 88])
class_b_scores = np.array([90, 87, 95, 82])

print(f"Class A scores: {class_a_scores}")
print(f"Class B scores: {class_b_scores}")

# Stack vertically (add Class B below Class A)
all_scores = np.vstack([class_a_scores, class_b_scores])
print(f"\nAll scores (classes as rows):\n{all_scores}")

# Stack horizontally (add Class B beside Class A)  
side_by_side = np.hstack([class_a_scores, class_b_scores])
print(f"All scores in one row: {side_by_side}")

# More complex example: Adding a new semester
semester1 = np.array([[85, 92], [78, 88]])  # 2 students × 2 tests
semester2 = np.array([[90, 87], [82, 95]])  # 2 students × 2 tests

print(f"\nSemester 1 scores:\n{semester1}")
print(f"Semester 2 scores:\n{semester2}")

# Combine semesters (add semester 2 as new columns)
yearly_scores = np.hstack([semester1, semester2])
print(f"\nFull year scores (4 tests per student):\n{yearly_scores}")
```

### Part E: Splitting Arrays - Breaking Data Apart

```python
print("\n=== Splitting Data for Analysis ===")

# Example: Hourly data for 24 hours
hourly_data = np.arange(24)  # Hours 0-23
print(f"24 hours of data: {hourly_data}")

# Split into 3 shifts of 8 hours each
shifts = np.split(hourly_data, 3)
morning, afternoon, night = shifts
print(f"Morning shift (0-7): {morning}")
print(f"Afternoon shift (8-15): {afternoon}")  
print(f"Night shift (16-23): {night}")

# Split at specific points (e.g., work hours vs off-hours)
work_split = np.split(hourly_data, [9, 17])  # Split at hour 9 and 17
before_work, work_hours, after_work = work_split
print(f"\nBefore work (0-8): {before_work}")
print(f"Work hours (9-16): {work_hours}")
print(f"After work (17-23): {after_work}")

# 2D splitting example
sales_matrix = np.arange(12).reshape(3, 4)  # 3 stores × 4 quarters
print(f"\nQuarterly sales by store:\n{sales_matrix}")

# Split by quarters
quarters = np.hsplit(sales_matrix, 4)  # Horizontal split
print(f"Q1 sales: {quarters[0].flatten()}")
print(f"Q4 sales: {quarters[3].flatten()}")
```

### Exercise
1. Given a list of test scores `[75, 82, 90, 68, 95, 78, 85]`, use boolean indexing to find scores above 80
2. Use fancy indexing to select the 1st, 3rd, and 5th elements from an array
3. Reshape a 1D array of 12 numbers into a 3×4 matrix, then into a 4×3 matrix
4. Create two 2×2 arrays and stack them both vertically and horizontally

### Answer

```python
# 1. Boolean indexing for high scores
scores = np.array([75, 82, 90, 68, 95, 78, 85])
print(f"All scores: {scores}")

high_scores = scores[scores > 80]
print(f"Scores above 80: {high_scores}")

# Find positions of high scores
high_score_positions = np.where(scores > 80)
print(f"Positions of high scores: {high_score_positions[0]}")

# 2. Fancy indexing for specific elements
data = np.array([10, 20, 30, 40, 50, 60])
indices = [0, 2, 4]  # 1st, 3rd, 5th (0-indexed)
selected = data[indices]
print(f"\nOriginal data: {data}")
print(f"Selected elements (1st, 3rd, 5th): {selected}")

# 3. Reshaping examples
numbers = np.arange(12)  # 0 to 11
print(f"\nOriginal 1D array: {numbers}")

# Reshape to 3×4
matrix_3x4 = numbers.reshape(3, 4)
print(f"Reshaped to 3×4:\n{matrix_3x4}")

# Reshape to 4×3
matrix_4x3 = numbers.reshape(4, 3)
print(f"Reshaped to 4×3:\n{matrix_4x3}")

# 4. Stacking arrays
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])
print(f"\nArray 1:\n{array1}")
print(f"Array 2:\n{array2}")

# Vertical stacking (top to bottom)
vertical_stack = np.vstack([array1, array2])
print(f"Stacked vertically:\n{vertical_stack}")

# Horizontal stacking (side by side)
horizontal_stack = np.hstack([array1, array2])
print(f"Stacked horizontally:\n{horizontal_stack}")
```

### Key Points
- Boolean indexing is perfect for filtering data based on conditions
- Fancy indexing lets you pick specific elements by their positions  
- Reshaping changes dimensions but keeps the same total number of elements
- Stacking combines multiple arrays - use `vstack` for rows, `hstack` for columns
- These techniques are essential for real-world data manipulation and analysis

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