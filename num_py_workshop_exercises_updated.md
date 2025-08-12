# NumPy Workshop: Hands-On Code, Exercises & Answers

> **INSTRUCTOR NOTES**: Welcome participants, do a quick round of introductions. Ask about their Python experience and what they hope to achieve with NumPy. Set expectations: this is hands-on, encourage questions, and emphasize that we'll progress from basics to practical applications.

## Overview

This workshop provides a comprehensive introduction to NumPy, the fundamental package for numerical computing in Python. Through hands-on coding exercises, you'll learn to work with arrays, perform vectorized operations, and apply NumPy to real-world data analysis problems.

> **INSTRUCTOR NOTES**: Emphasize that NumPy is the foundation of the entire scientific Python ecosystem. Mention that concepts learned here will transfer to pandas, scikit-learn, matplotlib, etc. Set the tone that this is practical, not theoretical.

*Examples and exercises are adapted from Jake VanderPlas's **Python Data Science Handbook** (Ch. 2: Introduction to NumPy), available at https://jakevdp.github.io/PythonDataScienceHandbook/*

> **INSTRUCTOR NOTES**: Credit Jake VanderPlas and encourage participants to explore the book for deeper learning. This adds credibility and provides a path for continued learning.

## Prerequisites

- Basic Python programming knowledge (variables, functions, loops)
- Familiarity with Python lists and basic data types
- A Python environment (Jupyter, IPython, or Python interpreter)

> **INSTRUCTOR NOTES**: Quickly check if anyone needs help with setup. Have a backup plan for participants with environment issues. Consider having them pair with someone or use a cloud environment like Google Colab.

## Workshop Schedule

> **INSTRUCTOR NOTES**: Walk through the schedule, emphasizing the progression from basic concepts to practical applications. Point out break times and mention that timing is flexible based on group pace. The mini-project at the end ties everything together.

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

> **INSTRUCTOR NOTES**: Explain the structure of each section. Emphasize that Questions help focus learning, Objectives set clear goals, Hands-On Code should be run together, Exercises are for practice, and Key Points summarize takeaways. Encourage participants to take notes on insights and questions.

Each section contains:
- **Questions**: Key learning questions to guide your understanding
- **Objectives**: What you'll be able to do after completing the section
- **Hands-On Code**: Minimal demos to run and explore
- **Exercises**: Short tasks to try individually or in pairs
- **Answers**: Expected results and complete solutions
- **Key Points**: Summary of essential concepts

---

## Document Overview

- **Introduction**: covers NumPy basics  
- **Exercises**: a set of practice problems  
- **Solutions**: example answers and explanations  



## 0. Welcome and icebreaker (09:00–09:15)

> **INSTRUCTOR NOTES**: Start by checking if everyone can access their Python environment. This is crucial - don't proceed if people can't run code. Use this time to build rapport and assess the group's experience level. Adjust your pace and depth based on their responses.

### Questions
- Is my Python environment properly configured for NumPy?
- How do I verify my NumPy installation and version?

> **INSTRUCTOR NOTES**: Have participants run this code together. It's a great way to ensure everyone's environment works and sets the collaborative tone. If someone has issues, pair them with a neighbor or provide alternative solutions (Google Colab, etc.).

### Objectives
After completing this setup, learners will be able to:
- Verify their Python and NumPy installation
- Import NumPy with the standard convention
- Run basic array operations to confirm the environment works

> **INSTRUCTOR NOTES**: Emphasize the `import numpy as np` convention - this is universal in the scientific Python community. Mention that version differences rarely matter for this workshop, but good to know what they're working with.

**Hands‑On Code**  
```python
import sys, numpy as np
print("Python version:", sys.version.split()[0])
print("NumPy version:", np.__version__)
```

> **INSTRUCTOR NOTES**: Walk around and check everyone's output. Typical versions should be Python 3.7+ and NumPy 1.19+. If someone has much older versions, warn them there might be minor differences but most content will work.

**Exercise**  
Run the code above. Confirm that you’re running Python 3.6+ and NumPy 1.15+ (or the latest installed), and that the test array prints correctly.  

**Answer**  
```text
Python version: 3.13.1
NumPy version: 2.3.1
```
Participants should see their local versions, confirming a valid environment.

> **INSTRUCTOR NOTES**: Point out that their versions may differ, and that's fine. Use this opportunity to transition into why NumPy is so important in the Python ecosystem.

### Key Points
- Use `import numpy as np` as the standard NumPy import convention
- Verify your environment with version checks before starting numerical work
- NumPy arrays are the foundation for all scientific Python computing

> **INSTRUCTOR NOTES**: These key points set up the next section perfectly. Emphasize that NumPy is the foundation - pandas DataFrames are built on NumPy arrays, scikit-learn uses NumPy arrays, etc. This motivates the entire workshop.

---

## 1. Why Numpy? (09:15–09:45)

> **INSTRUCTOR NOTES**: This is a crucial section that motivates the entire workshop. Many beginners wonder "why not just use Python lists?" Take time with the comparisons - they're eye-opening. Be prepared for questions about when to use each data structure.

### Questions
- Why should I use NumPy instead of Python lists?
- How much faster is NumPy for numerical computations?
- What makes NumPy arrays more memory efficient?

> **INSTRUCTOR NOTES**: Ask participants if they've worked with large datasets in pure Python. Many will have experienced slow performance or memory issues. Use this to create engagement - NumPy solves real problems they've likely encountered.

### Objectives
After completing this episode, learners will be able to:
- Compare performance between Python lists and NumPy arrays
- Explain the memory advantages of NumPy arrays  
- Demonstrate vectorized operations vs explicit loops
- Import NumPy using the standard convention

> **INSTRUCTOR NOTES**: Emphasize that understanding these differences is fundamental to making good design choices. After this section, they should never wonder "why NumPy?" again.

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

> **INSTRUCTOR NOTES**: Have everyone run this code first. Point out that `np.array()` is the most common way to create arrays. The type `numpy.ndarray` (n-dimensional array) is what makes everything else possible.

### NumPy ndarray vs Python Lists: Key Differences

Understanding the fundamental differences between NumPy arrays and Python lists is crucial for beginners. Let's explore this with common operations:

> **INSTRUCTOR NOTES**: This comparison section is the heart of the motivation. Take your time here - these examples are designed to be surprising and enlightening. Make sure everyone understands each operation before moving on.

#### Addition Operations

```python
print("=== Addition: Element-wise vs List Concatenation ===")

# Python lists: + means concatenation
python_list1 = [1, 2, 3]
python_list2 = [4, 5, 6]
list_addition = python_list1 + python_list2
print(f"Python lists: {python_list1} + {python_list2} = {list_addition}")
print("  Result: Lists are concatenated (joined together)")

# NumPy arrays: + means element-wise addition
numpy_array1 = np.array([1, 2, 3])
numpy_array2 = np.array([4, 5, 6])
array_addition = numpy_array1 + numpy_array2
print(f"NumPy arrays: {numpy_array1} + {numpy_array2} = {array_addition}")
print("  Result: Corresponding elements are added together")

# Adding a number to each element
print(f"\nAdding 10 to each element:")
try:
    list_plus_number = python_list1 + 10  # This will fail!
except TypeError as e:
    print(f"Python list + 10: Error - {e}")

array_plus_number = numpy_array1 + 10  # This works!
print(f"NumPy array + 10: {numpy_array1} + 10 = {array_plus_number}")
```

> **INSTRUCTOR NOTES**: This example often gets "aha!" moments. Python list + number fails because lists don't know how to add numbers to each element. NumPy arrays treat this as "broadcast this operation to every element" - a powerful concept we'll explore more later.

#### Multiplication Operations

```python
print(f"\n=== Multiplication: Repetition vs Element-wise ===")

# Python lists: * means repetition
list_multiplication = python_list1 * 3
print(f"Python list * 3: {python_list1} * 3 = {list_multiplication}")
print("  Result: List is repeated 3 times")

# NumPy arrays: * means element-wise multiplication
array_multiplication = numpy_array1 * 3
print(f"NumPy array * 3: {numpy_array1} * 3 = {array_multiplication}")
print("  Result: Each element is multiplied by 3")

# Multiplying two sequences element-wise
print(f"\nElement-wise multiplication:")
try:
    list_element_mult = python_list1 * python_list2  # This doesn't work as expected
    print(f"Python lists: Can't multiply element-wise directly")
except TypeError as e:
    print(f"Python lists: {e}")

array_element_mult = numpy_array1 * numpy_array2
print(f"NumPy arrays: {numpy_array1} * {numpy_array2} = {array_element_mult}")
print("  Result: Corresponding elements are multiplied")
```

> **INSTRUCTOR NOTES**: Point out how Python's * operator is overloaded differently for lists vs arrays. With lists it means "repeat", with NumPy it means "element-wise multiply". This is a common source of confusion for beginners coming from pure Python.

#### Mathematical Functions

```python
print(f"\n=== Mathematical Functions ===")

# Python lists: Need loops or list comprehensions
import math
python_squares = [x**2 for x in python_list1]  # Manual computation
python_sqrt = [math.sqrt(x) for x in python_list1]  # Manual computation
print(f"Python list squares: [x**2 for x in {python_list1}] = {python_squares}")
print(f"Python list sqrt: [math.sqrt(x) for x in {python_list1}] = {python_sqrt}")
print("  Result: Requires loops or comprehensions")

# NumPy arrays: Built-in vectorized functions
numpy_squares = numpy_array1 ** 2  # Vectorized operation
numpy_sqrt = np.sqrt(numpy_array1)  # Vectorized function
print(f"NumPy array squares: {numpy_array1} ** 2 = {numpy_squares}")
print(f"NumPy array sqrt: np.sqrt({numpy_array1}) = {numpy_sqrt}")
print("  Result: Direct vectorized operations")
```

> **INSTRUCTOR NOTES**: Show how NumPy's approach eliminates loops entirely. The `**` operator and `np.sqrt()` apply to every element automatically. This is called "vectorization" - a core NumPy concept.

#### Memory and Performance

```python
print(f"\n=== Memory and Performance Differences ===")

# Create larger datasets for comparison
large_list = list(range(1000))
large_array = np.arange(1000)

# Memory usage
import sys
list_memory = sys.getsizeof(large_list) + sum(sys.getsizeof(x) for x in large_list)
array_memory = sys.getsizeof(large_array)

print(f"Memory usage for 1000 integers:")
print(f"  Python list: ~{list_memory} bytes")
print(f"  NumPy array: {array_memory} bytes")
print(f"  NumPy uses ~{list_memory // array_memory}x less memory")

# Type homogeneity
mixed_list = [1, 2.5, "hello", True]  # Python allows mixed types
print(f"\nType flexibility:")
print(f"Python list: {mixed_list} (mixed types allowed)")
numpy_array_mixed = np.array([1, 2.5, 3, 4])  # NumPy promotes to common type
print(f"NumPy array: {numpy_array_mixed} (all converted to {numpy_array_mixed.dtype})")
```

> **INSTRUCTOR NOTES**: The memory comparison is usually shocking - NumPy can use 10x less memory! Also note how NumPy promotes mixed types to a common type (type promotion), while Python lists keep the original types.

#### Summary of Key Differences

**Python Lists:**
- ✓ Mixed data types (numbers, strings, objects)
- ✓ Dynamic sizing (append, insert, remove)
- ✓ General-purpose data structure
- ✗ No built-in mathematical operations
- ✗ Slower for numerical computations
- ✗ More memory overhead

**NumPy Arrays:**
- ✓ Fast mathematical operations
- ✓ Memory efficient
- ✓ Vectorized computations
- ✓ Built-in mathematical functions
- ✗ Fixed data type (homogeneous)
- ✗ Fixed size (less dynamic)

**When to Choose:**
- **Use Python Lists** for mixed data types, dynamic collections, and general programming tasks
- **Use NumPy Arrays** for numerical computations, scientific computing, and performance-critical operations

> **INSTRUCTOR NOTES**: Emphasize that both data structures have their place. Don't present this as "NumPy is always better" - each has optimal use cases. Lists are still the right choice for many programming tasks.

### Python Array Module vs NumPy Arrays

Python also has a built-in `array` module that sits between lists and NumPy arrays. Here's how they compare:

```python
import array
import numpy as np

# Creating arrays with different approaches
python_array = array.array('i', [1, 2, 3, 4, 5])  # 'i' = signed integer
numpy_array = np.array([1, 2, 3, 4, 5])

print("=== Python Array vs NumPy Array ===")
print(f"Python array: {python_array}")
print(f"NumPy array:  {numpy_array}")
```
```python
# Demonstrating the differences
print(f"\n=== Capabilities Comparison ===")

# Mathematical operations
try:
    # Python array: limited math support
    array_doubled = array.array('i', [x * 2 for x in python_array])  # Manual
    print(f"Python array * 2: {array_doubled} (manual operation)")
except:
    print("Python array: Limited mathematical operations")

# NumPy array: vectorized operations
numpy_doubled = numpy_array * 2  # Automatic vectorization
print(f"NumPy array * 2:  {numpy_doubled} (vectorized operation)")

# Memory efficiency comparison
import sys
print(f"\nMemory usage for 1000 integers:")
large_python_array = array.array('i', range(1000))
large_numpy_array = np.arange(1000, dtype=np.int32)

print(f"Python array: {sys.getsizeof(large_python_array)} bytes")
print(f"NumPy array:  {sys.getsizeof(large_numpy_array)} bytes")

# Multidimensional capability
print(f"\nMultidimensional support:")
print("Python array: 1D only")
numpy_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"NumPy array: Can be multidimensional\n{numpy_2d}")
```

> **INSTRUCTOR NOTES**: Most beginners don't know about Python's built-in array module. This comparison helps complete their understanding of the data structure landscape. Focus on the key point: array module is a middle ground, but NumPy's ecosystem makes it the clear choice for data work.

**Key Differences:**

| Feature | Python `array` | NumPy `ndarray` |
|---------|----------------|-----------------|
| **Dimensions** | 1D only | Multidimensional |
| **Math Operations** | Limited | Full vectorized support |
| **Memory** | More efficient than lists | Most efficient |
| **Dependencies** | Built-in Python | Requires NumPy library |
| **Type Control** | Manual type codes ('i', 'f', 'd') | Automatic type inference |
| **Performance** | Moderate | Fastest for numerical work |
| **Ecosystem** | Minimal | Huge scientific ecosystem |

**When to Use Python's `array` Module:**
- Simple 1D numeric data without NumPy dependency
- Memory efficiency more important than functionality
- Interfacing with C libraries requiring specific data layouts
- Binary data processing with strict type control

**When to Use NumPy Arrays:**
- Scientific computing and data analysis
- Mathematical operations on arrays
- Multidimensional data (matrices, tensors)
- Performance-critical numerical computations
- Integration with scientific Python ecosystem


> **INSTRUCTOR NOTES**: This reinforces the core concept - operations apply to every element automatically. No loops needed! This is the foundation that makes NumPy powerful.

### Why is NumPy Faster?

Let's compare doing the same task with Python lists vs NumPy arrays:
```python
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
```
> **INSTRUCTOR NOTES**: The performance difference is usually dramatic - often 10-100x faster! Explain that NumPy's speed comes from C implementations under the hood. For small arrays the difference matters less, but for real data work it's transformative.
```python

### When Should You Use NumPy?

```python
# NumPy is great for:
print("\n=== When to Use NumPy ===")

# Large amounts of numerical data due to less memory consumption and performance

# Mathematical operations on many numbers
scores = np.array([85, 92, 78, 96, 88])
print(f"\nTest scores: {scores}")
print(f"Add 5 bonus points to all: {scores + 5}")

temperatures = np.array([20.5, 22.1, 19.8, 25.0, 23.2])
print(f"Daily temperatures: {temperatures}")
print(f"Average temperature: {temperatures.mean():.1f}°C")
print(f"Hottest day: {temperatures.max():.1f}°C")

# Working with grids/tables of data
sales_data = np.array([[100, 150, 200],    # Store 1: Jan, Feb, Mar
                       [120, 130, 180],    # Store 2: Jan, Feb, Mar  
                       [90,  140, 220]])   # Store 3: Jan, Feb, Mar

print(f"\nSales data shape: {sales_data.shape} (3 stores × 3 months)")
print(f"Total sales per store: {sales_data.sum(axis=1)}")
print(f"Total sales per month: {sales_data.sum(axis=0)}")
```

> **INSTRUCTOR NOTES**: These examples show NumPy's practical value. Point out how `.mean()`, `.max()`, and `.sum()` are built-in methods that would require manual loops in pure Python. The sales_data example previews multidimensional arrays and the axis parameter.

### Exercise
1. Create a NumPy array with all even numbers from 2 to 100
2. Calculate the standard deviation of these numbers using the mathematical formula: σ = √(Σ(x - μ)²/N)


> **INSTRUCTOR NOTES**: This exercise demonstrates NumPy's power for both array creation and mathematical operations. The standard deviation calculation shows vectorized operations in action. Give them 8-10 minutes and encourage them to work through the formula step by step.

#### Answer

```python
import numpy as np
import time

# 1. Create array of even numbers from 2 to 100 using logic
print("=== Creating Even Numbers Array Using Logic ===")

even_numbers_list = [i for i in range(1, 101) if i % 2 == 0]
even_numbers = np.array(even_numbers_list)
print(f"Using list comprehension: {even_numbers[:10]}... (showing first 10)")

# 2. Calculate standard deviation using the formula: σ = √(Σ(x - μ)²/N)
print(f"\n=== Manual Standard Deviation Calculation ===")

# Step by step using NumPy vectorized operations
mean_value = even_numbers.mean()                    # μ (mu)
print(f"Mean (μ): {mean_value}")

deviations = even_numbers - mean_value              # (x - μ)
print(f"Deviations from mean (first 5): {deviations[:5]}")

squared_deviations = deviations ** 2                # (x - μ)²
print(f"Squared deviations (first 5): {squared_deviations[:5]}")

variance = squared_deviations.mean()                # Σ(x - μ)²/N
print(f"Variance (σ²): {variance}")

manual_std = np.sqrt(variance)                      # σ = √(variance)
print(f"Manual standard deviation: {manual_std:.6f}")

```

### Key Points
- NumPy arrays are significantly faster than Python lists for numerical operations
- NumPy provides vectorized operations that eliminate the need for explicit loops
- Memory usage is much more efficient with NumPy due to homogeneous data types
- NumPy operations are implemented in C, providing near-native performance
- Use `import numpy as np` as the standard convention

> **INSTRUCTOR NOTES**: End this section by asking if there are questions about when to use NumPy vs lists. This is a common point of confusion. Remind them that understanding these trade-offs will guide their design decisions throughout their careers in data science/scientific computing.

---

## 2. Creating ndarrays (09:45–10:30)

> **INSTRUCTOR NOTES**: This section moves from "why NumPy" to "how to use NumPy". Focus on the most common array creation methods first. Students often get overwhelmed by all the options, so emphasize np.array(), np.zeros(), np.ones(), and np.arange() as the core methods.

### Questions
- How do I create NumPy arrays with different initialization patterns?
- What's the difference between np.zeros, np.ones, np.empty, and np.full?
- How do I inspect array properties like shape, dtype, and memory usage?

> **INSTRUCTOR NOTES**: These questions address the most common beginner needs. Many students come from MATLAB or other environments and expect similar functions - reassure them that NumPy has all the standard array creation tools.

### Objectives
After completing this episode, learners will be able to:
- Create arrays using various NumPy creation functions
- Explain the differences between different array initialization methods
- Inspect array attributes and understand their memory implications
- Choose the appropriate creation method for different use cases

> **INSTRUCTOR NOTES**: The learning objectives cover the essential skills for array creation. Emphasize that choosing the right creation method depends on your use case - sometimes you have data (use np.array), sometimes you need initialized arrays (zeros/ones), sometimes you need sequences (arange/linspace).

### Basic Array Creation Functions

```python
import numpy as np

# Creating arrays from Python lists
list_array = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"From list: {list_array}")
print(f"2D array:\n{matrix}")
print(f"  .shape: {list_array.shape} - Dimensions")
print(f"  .dtype: {list_array.dtype} - Data type") 
```
```python
%pip install tifffile
import tifffile as tiff

# Read the TIFF file
img = tiff.imread("B03.tif")
arr = np.array(img)

print("Shape:", arr.shape)
print("Data type:", arr.dtype)
print("Contents:\n", arr)
```

### Datatypes
![datatype](https://numpy.org/doc/stable/_images/dtype-hierarchy.png)

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
```


> **INSTRUCTOR NOTES**: Start with `np.array()` since it's the most intuitive - converting existing Python data. Point out how nested lists become 2D arrays automatically. This is the gateway function that gets people started with NumPy.

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

> **INSTRUCTOR NOTES**: zeros() and ones() are workhorses for initialization. Emphasize the shape parameter - single number for 1D, tuple for multidimensional. The dtype parameter is crucial - many beginners forget this and get float64 when they wanted integers. full() is less common but shows the pattern.

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

> **INSTRUCTOR NOTES**: arange vs linspace is a common source of confusion. Emphasize: arange is like Python's range (specify step), linspace is for when you know how many points you want. Show them the practical difference - arange might not include the endpoint due to floating point precision, linspace always includes it (unless endpoint=False).

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

> **INSTRUCTOR NOTES**: Emphasize the modern random API (default_rng) vs legacy numpy.random functions. The seeded generator ensures reproducible results for teaching. Identity matrices (np.eye) are essential for linear algebra. Show how random arrays are useful for testing and simulation.

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

> **INSTRUCTOR NOTES**: Array inspection is crucial for debugging and understanding performance. Point out that shape tells you dimensions, size tells you total elements, and nbytes shows memory usage. These attributes are essential for working with large datasets.

### Data Type Information

```python

# Data type drawbacks and limitations
print(f"\n=== Data Type Drawbacks and Limitations ===")
import sys  # For memory size calculations

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

print(f"   1M elements as float32: {sys.getsizeof(large_float32) / 1024**2:.1f} MB")
print(f"   1M elements as float64: {sys.getsizeof(large_float64) / 1024**2:.1f} MB")
print(f"   Memory savings with float32: {(1 - sys.getsizeof(large_float32)/sys.getsizeof(large_float64))*100:.0f}%")

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

> **INSTRUCTOR NOTES**: The data types section is crucial but can be overwhelming. Focus on practical implications: int8/16 can overflow, float32 saves memory but loses precision, uint types can't be negative. Many students haven't thought about these trade-offs before. Emphasize that NumPy forces you to be explicit about these choices.

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
1. Create a 3×4 array filled with random values, then report min, max, and mean
2. Create a 5×5 identity matrix and modify it to have 2s on the diagonal  
3. Use `np.linspace()` to create 50 evenly spaced points between 0 and 2π, then compute sin of each point
4. Compare `np.arange(0, 1, 0.1)` vs `np.linspace(0, 1, 11)` - what's the difference?
```
> **INSTRUCTOR NOTES**: Give them 8-10 minutes. This exercise covers the core array creation functions and introduces some basic operations. Question 4 often reveals the arange vs linspace confusion - use this as a teaching moment.

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

> **INSTRUCTOR NOTES**: Wrap up by emphasizing that array creation is foundational - getting the right shape and dtype at creation saves time later. Preview that next we'll learn how to access and modify array elements through indexing.

## 3. Indexing, slicing and views (10:45–12:15)

> **INSTRUCTOR NOTES**: Indexing is where many students get confused, especially coming from Python lists. Take time with multidimensional indexing and views vs copies. The broadcasting concepts here set up the universal functions section. This is a long section - consider a 10-minute break halfway through.

### Questions
- How do I access and modify specific elements or sections of arrays?
- What's the difference between views and copies when slicing arrays?
- How can I use boolean and fancy indexing for data filtering and selection?

> **INSTRUCTOR NOTES**: These questions address the core confusion points. Many students struggle with the view/copy distinction and don't initially see the power of boolean indexing for data filtering.

### Objectives
After completing this episode, learners will be able to:
- Use basic indexing and slicing to access array elements and subarrays
- Apply boolean indexing for conditional data selection
- Understand the difference between views and copies in array operations
- Use fancy indexing for complex array manipulations

> **INSTRUCTOR NOTES**: Start with a clear example array that everyone can follow. The reshape is introduced here - mention it briefly but don't dive deep, focus on the indexing concepts.


**Hands‑On Code**  
```python

# Quick reshape demo
print("\n=== Reshape: Changes dimensions, keeps data ===")
data = np.arange(6)              # [0, 1, 2, 3, 4, 5]
matrix = data.reshape(2, 3)      # [[0, 1, 2], [3, 4, 5]]
auto = data.reshape(-1, 2)       # Auto-calculate: (3, 2)
print(f"1D: {data}")
print(f"2×3:\n{matrix}")
print(f"Auto (-1,2):\n{auto}")

import numpy as np
x = np.arange(12).reshape(3,4)  # reshape(3,4) turns 12 elements into 3×4 matrix
print("Original array:")
print(x)

# Basic indexing
print(f"\nBasic indexing:")
print(f"x[0,1] = {x[0,1]}")  # single element
print(f"x[2,3] = {x[2,3]}")  # single element
print(f"x[-1,-1] = {x[-1,-1]}")  # negative indices

# Slicing
print(f"\nSlicing:")
```
> **INSTRUCTOR NOTES**: Point out that negative indices work just like Python lists. The comma syntax `x[row, col]` is NumPy-specific and much cleaner than `x[row][col]`. Make sure everyone understands this notation before moving on.
```python
print(f"First row: {x[0, :]}")
print(f"Last column: {x[:, -1]}")
print(f"First 2 rows, last 2 columns:\n{x[:2, -2:]}")
print(f"Every other row: \n{x[::2, :]}")
print(f"Reverse rows:\n{x[::-1, :]}")

# Integer array indexing (fancy indexing)
```
> **INSTRUCTOR NOTES**: The slicing syntax is exactly like Python lists, but applied to each dimension. The `::2` step syntax and negative steps (`[::-1]`) are powerful but can confuse beginners. Spend time on the multi-dimensional slicing - this is where NumPy shines over nested lists.
```python
rows = np.array([0, 2])
cols = np.array([1, 3])
print(f"\nFancy indexing:")
print(f"x[{rows}, {cols}] = {x[rows, cols]}")  # picks (0,1) and (2,3)

# More complex fancy indexing
print(f"Multiple row selection:\n{x[[0, 2], :]}")  # Select rows 0 and 2
print(f"Multiple col selection:\n{x[:, [1, 3]]}")  # Select columns 1 and 3

# Boolean indexing
```
> **INSTRUCTOR NOTES**: Fancy indexing (using arrays of indices) is powerful but can be confusing. Emphasize that `x[[0,2], :]` selects rows 0 and 2, while `x[:, [1,3]]` selects columns 1 and 3. This is different from slicing - you're picking specific rows/columns, not ranges.
```python
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
```
> **INSTRUCTOR NOTES**: Boolean indexing is one of NumPy's killer features for data analysis. Point out that `x % 2 == 0` creates a boolean array of the same shape, then `x[mask]` selects only the True elements. The `&` and `|` operators (not `and`/`or`) are essential for combining conditions. This is the foundation of data filtering in pandas.
```python
print(f"\nModifying arrays:")
```

### Views vs Copies: Rules of Thumb and Examples

Understanding when NumPy creates views (shares memory) vs copies (new memory) is crucial for both performance and avoiding bugs. Here are the essential rules with examples:

> **INSTRUCTOR NOTES**: This is where many students get confused and make bugs later. Take time with this section! The view/copy distinction is fundamental to NumPy programming. Use the gradebook example to make it practical - everyone understands student grades. Have students run np.shares_memory() themselves to see the difference.

```python
# Reset to clean array for demonstrations
x = np.arange(24).reshape(4, 6)
```

#### Rule 1: Basic Slicing → Views (Shares Memory)
```python
slice_view = x[1:3, 2:5]      # Rectangular slice
row_view = x[2, :]            # Entire row  
col_view = x[:, 3]            # Entire column
step_view = x[::2, ::2]       # Stepped slice

# All are views - they share memory with original
print(f"slice_view shares memory: {np.shares_memory(x, slice_view)}")  # True

# Modifying view affects original
slice_view[:] = -999
print("Original array was modified!")
```

#### Rule 2: Fancy Indexing → Copies (Independent Memory)
```python
fancy_copy = x[[0, 2], :]              # Non-contiguous rows
fancy_copy2 = x[:, [1, 3, 5]]          # Non-contiguous columns
fancy_copy3 = x[[0, 2], [1, 3]]        # Specific elements

# All are copies - independent memory
print(f"fancy_copy shares memory: {np.shares_memory(x, fancy_copy)}")  # False

# Modifying copy doesn't affect original
fancy_copy[:] = -777
print("Original array unchanged!")
```

#### Rule 3: Boolean Indexing → Copies (Always Safe)  
```python
mask = x > 10
bool_copy = x[mask]

# Boolean indexing always creates copies
print(f"bool_copy shares memory: {np.shares_memory(x, bool_copy)}")  # False
```

#### Rule 4: Force Copies When Needed
```python
# When you need a safe copy from slicing
safe_copy = x[1:3, :].copy()  # Explicitly make copy
safe_copy[:] = -111           # Won't affect original
```

**Quick Reference:**
- **Views:** Basic slicing `arr[start:stop]`, single elements `arr[i,j]`
- **Copies:** Fancy indexing `arr[[0,2]]`, boolean indexing `arr[mask]`, explicit `.copy()`
- **Check:** Use `np.shares_memory(original, result)` to verify
```python
# Common gotcha: Modifying a slice unintentionally
data = np.arange(20).reshape(4, 5)
print(f"Original data:\n{data}")

# This modifies the original!
subset = data[1:3, :]  # This is a view
subset += 1000         # This modifies the original data!
print(f"After modifying subset, original data:\n{data}")
print("⚠️  Gotcha! Adding to the view modified the original!")

# Reset and show safe approach
data = np.arange(20).reshape(4, 5)
subset_safe = data[1:3, :].copy()  # Safe copy
subset_safe += 1000                 # Only modifies the copy
print(f"After modifying subset_safe copy, original data:\n{data}")
print("✅ Safe! Used .copy() to protect original.")

import sys  # For memory size calculations

# Memory comparison
large_array = np.random.rand(1000, 1000)
slice_view = large_array[::10, ::10]    # View - no new memory
slice_copy = large_array[::10, ::10].copy()  # Copy - new memory

print(f"Original array: {sys.getsizeof(large_array) / 1024**2:.1f} MB")
print(f"Slice view: {sys.getsizeof(slice_view) / 1024**2:.1f} MB (shares memory)")
print(f"Slice copy: {sys.getsizeof(slice_copy) / 1024**2:.1f} MB (independent memory)")
print(f"Memory overhead for copy: {sys.getsizeof(slice_copy) / 1024**2:.1f} MB")

print(f"\n" + "="*60)
print("QUICK REFERENCE: View vs Copy Rules")
print("="*60)
print("""
VIEWS (shares memory - fast, but changes affect original):
✓ Basic slicing: arr[start:stop:step, start:stop:step]
✓ Single element access: arr[i, j] 
✓ Full dimensions: arr[:, :], arr[i, :], arr[:, j]
✓ Regular steps: arr[::2, ::3]

COPIES (independent memory - safe, but uses more memory):
✓ Fancy indexing: arr[[0,2,4], :], arr[:, [1,3,5]]
✓ Boolean indexing: arr[arr > 5], arr[mask]
✓ Explicit copy: arr.copy(), arr[:].copy()
✓ Some operations: arr.reshape(-1) (sometimes)

WHEN TO USE WHICH:
→ Use views when: You want to modify original data, memory is limited
→ Use copies when: You want independence, safety from side effects

QUICK CHECK: Use np.shares_memory(original, result) to verify!
""")

# Modifying arrays through indexing
print(f"\nModifying arrays:")
# Slice view and modify
x = np.arange(12).reshape(3,4)
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
```
### Advanced Indexing Patterns

> **INSTRUCTOR NOTES**: The gradebook example makes indexing practical and relatable. Students can see immediate applications to real data analysis. Point out how these patterns apply to any tabular data - sales, measurements, survey responses. The "gotchas" section prevents common bugs that plague beginners.

#### Practical Example: Student Gradebook
```python
# Sample data: 5 students × 4 subjects
grades = np.array([
    [85, 92, 78, 88],  # Student 0: Math, Science, English, History
    [90, 85, 82, 91],  # Student 1
    [78, 88, 85, 79],  # Student 2  
    [92, 91, 89, 93],  # Student 3
    [76, 82, 80, 84]   # Student 4
])
subjects = ['Math', 'Science', 'English', 'History']


#### Pattern 1: Multi-Condition Selection
# Students with ALL grades ≥ 80
good_students = np.all(grades >= 80, axis=1)
good_grades = grades[good_students, :]

# Students who scored >90 in Math OR >87 overall average
math_stars = grades[:, 0] > 90
high_achievers = np.mean(grades, axis=1) > 87
special = math_stars | high_achievers
special_students = grades[special, :]


#### Pattern 2: Conditional Replacement

# Add 5 bonus points to grades < 80
grades_bonus = grades.copy()
grades_bonus[grades_bonus < 80] += 5

# Cap maximum at 95
grades_bonus[grades_bonus > 95] = 95


#### Pattern 3: Finding Positions

# Find highest grade position
max_pos = np.unravel_index(np.argmax(grades), grades.shape)
print(f"Highest grade at student {max_pos[0]}, subject {max_pos[1]}")

# Find all positions with specific grade
positions_91 = np.where(grades == 91)
```

#### Common Gotchas and Safe Patterns
```python
# ❌ WRONG: Accidentally modifying original
subset = grades[:2, :]  # This is a view!
subset += 10           # Modifies original grades!
```

#### ✅ RIGHT: Safe modification
```python
subset_safe = grades[:2, :].copy()  # Explicit copy
subset_safe += 10                   # Only modifies copy
```

#### ✅ RIGHT: Direct assignment
```python
grades_modified = grades.copy()
grades_modified[:2, :] += 10       # Clear intent
```

**Best Practices:**
- **Views:** Use for read-only operations, memory-limited situations
- **Copies:** Use when modifying subsets, need independence
- **Conditions:** Use `&`, `|` (not `and`, `or`), add parentheses: `(a > 5) & (b < 10)`
- **Debugging:** Check with `np.shares_memory()` when unsure

#### Advanced indexing patterns
```python
x = np.arange(24).reshape(4, 6)
print(f"\nAdvanced patterns on 4x6 array:")
print(f"Original:\n{x}")
```
#### Diagonal elements
```python
diag_indices = np.arange(min(x.shape))
print(f"Diagonal: {x[diag_indices, diag_indices]}")
```
#### Block selection
```python
print(f"2x2 blocks (top-left and bottom-right):")
print(f"Top-left 2x2:\n{x[:2, :2]}")
print(f"Bottom-right 2x2:\n{x[-2:, -2:]}")
```
#### Conditional replacement
```python
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

> **INSTRUCTOR NOTES**: This section covers a lot of ground! Make sure students understand the view/copy distinction - it's crucial for memory management and avoiding bugs. Boolean indexing is the gateway to data analysis workflows. Consider taking a 10-15 minute break here before Universal Functions.

---

## 4. Universal functions and aggregations (13:15–14:45)

> **INSTRUCTOR NOTES**: This section demonstrates NumPy's real power for data analysis. Universal functions (ufuncs) are what make NumPy fast and convenient. The presidential heights data makes it practical and engaging. This is where students see how NumPy enables real data analysis workflows.

### Questions
- What are universal functions (ufuncs) and how do they work?
- What is broadcasting and how does it enable operations between arrays of different shapes?
- How do vectorized operations compare to traditional loops in performance?
- How can I apply functions element-wise across arrays efficiently?

> **INSTRUCTOR NOTES**: Focus on the "vectorized" concept - operations that work on entire arrays without explicit loops. This is the paradigm shift that makes NumPy powerful. The aggregation functions connect directly to what data scientists need daily.

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

> **INSTRUCTOR NOTES**: This exercise combines universal functions with real data analysis. Give them 10-12 minutes. The president heights question connects everything together - loading data, computing statistics, and answering a real question. This is what NumPy enables that pure Python makes difficult.

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

> **INSTRUCTOR NOTES**: Universal functions are the heart of NumPy's performance advantage. Make sure students understand that "vectorized" doesn't mean using loops - it means the loops are implemented in C under the hood. This is the key insight that makes NumPy transformative for numerical computing.

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

> **INSTRUCTOR NOTES**: Students are getting tired by now - this is late afternoon content. Keep it practical and hands-on. Focus on reshape and transpose as the most immediately useful operations. Many students struggle with mental models of multidimensional arrays, so use visual analogies.

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

**Reshape changes array dimensions while keeping the same data.** Use it to reorganize data for different analysis perspectives.

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

> **INSTRUCTOR NOTES**: This is the culminating section where students apply everything they've learned. The performance comparison should be eye-opening. Give them time to experiment with the mini-project - this is where the concepts solidify. Many will want to work in pairs, which is fine.

### Questions
- How can I optimize NumPy code for better performance?
- What memory layout considerations affect array operations?
- How do I apply NumPy skills to solve real-world problems?

### Objectives
After completing this episode, learners will be able to:
- Apply performance optimization techniques for NumPy operations
- Understand memory layout and its impact on performance
- Complete a mini-project using NumPy skills learned throughout the workshop

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Row_and_column_major_order.svg/250px-Row_and_column_major_order.svg.png)


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

# Understanding memory order with np.nditer()
print(f"\n=== Understanding Memory Order with np.nditer() ===")
print("np.nditer() shows how NumPy traverses arrays in memory")

# Create a 2x3 array for demonstration
demo_array = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Demo array:\n{demo_array}")

# C-order (row-major) - default NumPy behavior
print(f"\nC-order (row-major) traversal:")
print("Reads row by row: [1,2,3] then [4,5,6]")
c_order = [x for x in np.nditer(demo_array, order='C')]
print(f"C-order iteration: {c_order}")

# F-order (column-major) - Fortran/MATLAB style
print(f"\nF-order (column-major) traversal:")
print("Reads column by column: [1,4] then [2,5] then [3,6]")
f_order = [x for x in np.nditer(demo_array, order='F')]
print(f"F-order iteration: {f_order}")

# Demonstrate with reshape to show memory layout impact
print(f"\n=== Memory Layout Impact on Operations ===")
large_array = np.arange(6).reshape(2, 3)
print(f"Array:\n{large_array}")

# Flatten with different orders
flat_c = large_array.flatten(order='C')  # Row-major
flat_f = large_array.flatten(order='F')  # Column-major
print(f"Flatten C-order: {flat_c}")  # [0,1,2,3,4,5]
print(f"Flatten F-order: {flat_f}")  # [0,3,1,4,2,5]

# Performance implications
print(f"\n=== Why Order Matters for Performance ===")
print("C-order (row-major): Better for row operations")
print("F-order (column-major): Better for column operations")
print("Choose order based on your data access patterns!")

print(f"\n=== Structured Data: Beyond Simple Arrays ===")

# Creating structured arrays for heterogeneous data
print(f"\n1. Basic Structured Arrays")
print("Useful when you need different data types in one array")

# Define a structured data type
student_dtype = np.dtype([
    ('name', 'U10'),        # Unicode string, max 10 chars
    ('age', 'i4'),          # 32-bit integer
    ('gpa', 'f8'),          # 64-bit float
    ('graduated', '?')      # Boolean
])

# Create structured array
students = np.array([
    ('Alice', 22, 3.8, True),
    ('Bob', 21, 3.2, False),
    ('Charlie', 23, 3.9, True),
    ('Diana', 20, 3.5, False)
], dtype=student_dtype)

print(f"Structured array:\n{students}")
print(f"Data type: {students.dtype}")

# Accessing fields like a database
print(f"\n2. Field Access (Like Database Columns)")
print(f"All names: {students['name']}")
print(f"All GPAs: {students['gpa']}")
print(f"Students who graduated: {students[students['graduated']]['name']}")

# Advanced structured arrays
print(f"\n3. Advanced Features")
# Compound operations
high_achievers = students[(students['gpa'] > 3.7) & (students['age'] > 21)]
print(f"High achievers over 21:\n{high_achievers}")

# Statistics on fields
print(f"Average GPA: {students['gpa'].mean():.2f}")
print(f"Age range: {students['age'].min()} - {students['age'].max()}")

# Record arrays (more convenient access)
print(f"\n4. Record Arrays (Attribute-Style Access)")
student_records = students.view(np.recarray)
print(f"Access with dot notation:")
print(f"First student name: {student_records.name[0]}")
print(f"All GPAs: {student_records.gpa}")

# Creating from dictionaries (practical approach)
print(f"\n5. Creating from Real Data")
# Simulating data that might come from CSV or database
data_dict = {
    'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'temperature': [23.5, 25.1, 22.8],
    'humidity': [65, 70, 68],
    'sensor_id': [1, 1, 2]
}

# Convert to structured array
sensor_dtype = np.dtype([
    ('timestamp', 'U10'),
    ('temperature', 'f4'),
    ('humidity', 'i2'),
    ('sensor_id', 'i2')
])

sensor_data = np.array(list(zip(*data_dict.values())), dtype=sensor_dtype)
print(f"Sensor data:\n{sensor_data}")

# Why use structured arrays?
print(f"\n=== When to Use Structured Arrays ===")
print("✓ Mixed data types in single array")
print("✓ Memory efficiency vs separate arrays")
print("✓ Database-like operations")
print("✓ Reading structured file formats")
print("✗ Performance overhead vs homogeneous arrays")
print("✗ Limited mathematical operations")
print("Note: For most data analysis, prefer pandas DataFrames")



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

> **INSTRUCTOR NOTES**: End on a high note! Students should feel accomplished and motivated to continue learning. The ecosystem overview shows them the broader picture - NumPy is the foundation for everything else they'll do in data science. Be encouraging about their progress and provide clear next steps.

### Summary
- NumPy arrays: creation, indexing, views, and memory
- Universal functions, aggregations, and performance
- Advanced indexing, reshaping, and real-world applications

> **INSTRUCTOR NOTES**: Quickly recap the journey - from "why NumPy" to practical data analysis. Emphasize that they now have the foundation for pandas, scikit-learn, matplotlib, and the entire scientific Python ecosystem. Ask for final questions and wish them well on their data science journey.

### Further Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [SciPy Lecture Notes](https://scipy-lectures.org/)
- [Awesome NumPy](https://github.com/hamelsmu/awesome-numpy)

### Feedback & Q&A
- What did you find most useful?
- Any remaining questions or topics?
- Suggestions for future workshops?

> **INSTRUCTOR NOTES**: Take time for genuine feedback and questions. Many students will have "aha" moments during this reflection time. Encourage them to share what clicked for them - it helps reinforce learning for everyone. End with enthusiasm about their NumPy journey!

---