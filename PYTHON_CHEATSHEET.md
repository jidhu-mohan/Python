# Python Cheatsheet

## Basic Syntax

### Variables and Assignment
```python
x = 5                    # Integer
y = 3.14                 # Float
name = "Python"          # String
is_valid = True          # Boolean
```

### Comments
```python
# Single line comment
"""
Multi-line
comment
"""
```

## Data Types

### Numbers
```python
int_num = 42             # Integer
float_num = 3.14         # Float
complex_num = 2 + 3j     # Complex

# Basic arithmetic
+ - * /                  # Addition, subtraction, multiplication, division
//                       # Floor division
%                        # Modulus (remainder)
**                       # Exponentiation
```

### Strings
```python
# String creation
single = 'Hello'
double = "World"
multi = """Multi
line string"""

# String methods
text = "python programming"
text.upper()             # "PYTHON PROGRAMMING"
text.lower()             # "python programming"
text.title()             # "Python Programming"
text.replace('python', 'java')  # "java programming"
text.split()             # ['python', 'programming']
len(text)                # 18

# String formatting
name = "Alice"
age = 30
f"My name is {name} and I'm {age}"        # f-strings (Python 3.6+)
"My name is {} and I'm {}".format(name, age)  # .format() method
"My name is %s and I'm %d" % (name, age)      # % formatting
```

## Data Structures

### Lists
```python
# Creation
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# Methods
numbers.append(6)        # Add to end
numbers.insert(0, 0)     # Insert at index
numbers.remove(3)        # Remove first occurrence
numbers.pop()            # Remove and return last item
numbers.pop(0)           # Remove and return item at index

# Slicing
numbers[0]               # First item
numbers[-1]              # Last item
numbers[1:4]             # Items from index 1 to 3
numbers[:3]              # First 3 items
numbers[2:]              # Items from index 2 to end
numbers[::2]             # Every 2nd item
```

### Dictionaries
```python
# Creation
person = {"name": "John", "age": 30, "city": "New York"}
person = dict(name="John", age=30, city="New York")

# Access and modify
person["name"]           # "John"
person.get("age")        # 30
person["email"] = "john@email.com"  # Add new key-value
person.update({"phone": "123-456"}) # Add multiple

# Methods
person.keys()            # Dict keys
person.values()          # Dict values
person.items()           # Key-value pairs
person.pop("age")        # Remove and return value
```

### Tuples
```python
# Creation (immutable)
coordinates = (10, 20)
point = 10, 20, 30       # Parentheses optional

# Access
coordinates[0]           # 10
x, y = coordinates       # Unpacking
```

### Sets
```python
# Creation (unique elements)
numbers = {1, 2, 3, 4, 5}
unique = set([1, 2, 2, 3, 3])  # {1, 2, 3}

# Methods
numbers.add(6)           # Add element
numbers.remove(3)        # Remove element (error if not found)
numbers.discard(10)      # Remove element (no error if not found)

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1 | set2              # Union: {1, 2, 3, 4, 5}
set1 & set2              # Intersection: {3}
set1 - set2              # Difference: {1, 2}
```

## Control Flow

### Conditional Statements
```python
if condition:
    # code block
elif another_condition:
    # code block
else:
    # code block

# Ternary operator
result = value_if_true if condition else value_if_false
```

### Loops
```python
# For loop
for item in iterable:
    print(item)

for i in range(5):       # 0 to 4
    print(i)

for i in range(1, 6):    # 1 to 5
    print(i)

for i in range(0, 10, 2): # 0 to 8, step 2
    print(i)

# While loop
while condition:
    # code block

# Loop control
break                    # Exit loop
continue                 # Skip to next iteration
```

### List Comprehensions
```python
# Basic syntax: [expression for item in iterable if condition]
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

## Functions

### Basic Functions
```python
def function_name(parameters):
    """Docstring (optional)"""
    # function body
    return value

# Examples
def greet(name):
    return f"Hello, {name}!"

def add(a, b=0):         # Default parameter
    return a + b

def multiply(*args):     # Variable arguments
    result = 1
    for num in args:
        result *= num
    return result

def info(**kwargs):      # Keyword arguments
    for key, value in kwargs.items():
        print(f"{key}: {value}")
```

### Lambda Functions
```python
# Syntax: lambda arguments: expression
square = lambda x: x**2
add = lambda x, y: x + y

# Often used with map, filter, sorted
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

## Built-in Functions

### Common Built-ins
```python
len(sequence)            # Length
max(iterable)            # Maximum value
min(iterable)            # Minimum value
sum(iterable)            # Sum of numbers
abs(number)              # Absolute value
round(number, digits)    # Round to n digits

# Type conversion
int(value)               # Convert to integer
float(value)             # Convert to float
str(value)               # Convert to string
list(iterable)           # Convert to list
tuple(iterable)          # Convert to tuple
set(iterable)            # Convert to set

# Useful functions
enumerate(iterable)      # Returns (index, value) pairs
zip(iter1, iter2)        # Combine iterables
sorted(iterable)         # Return sorted list
reversed(sequence)       # Return reversed iterator
```

## File Handling

```python
# Reading files
with open('filename.txt', 'r') as file:
    content = file.read()           # Read entire file
    lines = file.readlines()        # Read all lines as list
    for line in file:               # Iterate through lines
        print(line.strip())

# Writing files
with open('filename.txt', 'w') as file:
    file.write("Hello World")
    file.writelines(["Line 1\n", "Line 2\n"])

# File modes: 'r' (read), 'w' (write), 'a' (append), 'r+' (read/write)
```

## Error Handling

```python
try:
    # Code that might raise an exception
    result = 10 / 0
except ZeroDivisionError:
    # Handle specific exception
    print("Cannot divide by zero!")
except Exception as e:
    # Handle any other exception
    print(f"An error occurred: {e}")
else:
    # Runs if no exception occurred
    print("Success!")
finally:
    # Always runs
    print("Cleanup code")
```

## Classes and Objects

```python
class ClassName:
    # Class variable
    class_var = "shared"

    def __init__(self, param):
        # Constructor
        self.instance_var = param

    def method(self):
        # Instance method
        return self.instance_var

    @classmethod
    def class_method(cls):
        # Class method
        return cls.class_var

    @staticmethod
    def static_method():
        # Static method
        return "static"

# Usage
obj = ClassName("value")
obj.method()
ClassName.class_method()
ClassName.static_method()
```

## Modules and Packages

```python
# Import entire module
import math
math.sqrt(16)

# Import specific functions
from math import sqrt, pi
sqrt(16)

# Import with alias
import numpy as np
np.array([1, 2, 3])

# Import all (not recommended)
from math import *
```

## Useful Libraries (Import Examples)

```python
# Standard library
import os                # Operating system interface
import sys               # System parameters
import datetime          # Date and time
import json              # JSON handling
import re                # Regular expressions
import random            # Random numbers
import collections       # Specialized containers

# Popular third-party libraries
import numpy as np       # Numerical computing
import pandas as pd      # Data analysis
import matplotlib.pyplot as plt  # Plotting
import requests          # HTTP requests
```

## String Methods Quick Reference

```python
s = "Hello World"
s.lower()                # "hello world"
s.upper()                # "HELLO WORLD"
s.title()                # "Hello World"
s.strip()                # Remove whitespace
s.replace("World", "Python")  # "Hello Python"
s.split()                # ["Hello", "World"]
s.find("World")          # 6 (index)
s.count("l")             # 3
s.startswith("Hello")    # True
s.endswith("World")      # True
s.isdigit()             # False
s.isalpha()             # False (space)
s.isalnum()             # False (space)
```

## List Methods Quick Reference

```python
lst = [1, 2, 3]
lst.append(4)            # Add to end
lst.insert(0, 0)         # Insert at index
lst.extend([5, 6])       # Add multiple items
lst.remove(2)            # Remove first occurrence
lst.pop()                # Remove and return last
lst.pop(0)               # Remove and return at index
lst.index(3)             # Find index of value
lst.count(1)             # Count occurrences
lst.sort()               # Sort in place
lst.reverse()            # Reverse in place
lst.clear()              # Remove all items
```

## Dictionary Methods Quick Reference

```python
d = {"a": 1, "b": 2}
d.get("a")               # Get value (None if not found)
d.keys()                 # Dict keys
d.values()               # Dict values
d.items()                # Key-value pairs
d.update({"c": 3})       # Add/update multiple
d.pop("b")               # Remove and return value
d.popitem()              # Remove and return last item
d.clear()                # Remove all items
```

## Comparison and Logical Operators

```python
# Comparison
==  !=  <  >  <=  >=     # Equal, not equal, less/greater than
is  is not               # Identity comparison
in  not in               # Membership testing

# Logical
and  or  not             # Logical operators

# Examples
x in [1, 2, 3]          # True if x is 1, 2, or 3
x is None               # True if x is None
not x                   # Logical NOT
```

## Common Patterns

### Swapping Variables
```python
a, b = b, a
```

### Multiple Assignment
```python
x, y, z = 1, 2, 3
```

### Check if List is Empty
```python
if not my_list:         # Pythonic way
    print("List is empty")
```

### Iterate with Index
```python
for i, value in enumerate(items):
    print(f"{i}: {value}")
```

### Dictionary Iteration
```python
for key, value in dictionary.items():
    print(f"{key}: {value}")
```

### Join List to String
```python
words = ["Hello", "World"]
sentence = " ".join(words)  # "Hello World"
```

### Remove Duplicates
```python
unique_items = list(set(items))
```

This cheatsheet covers the most essential Python concepts for quick reference!