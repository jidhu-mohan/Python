# Pandas Cheatsheet

## Import Pandas
```python
import pandas as pd
import numpy as np
```

## Data Structures

### Series (1D)
```python
# Create Series
s = pd.Series([1, 2, 3, 4, 5])
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# Series properties
s.values                      # Underlying numpy array
s.index                       # Index labels
s.dtype                       # Data type
s.shape                       # Dimensions
```

### DataFrame (2D)
```python
# Create DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df = pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=['A', 'B'])
df = pd.DataFrame(np.random.randn(4, 3), columns=['A', 'B', 'C'])

# DataFrame properties
df.shape                      # (rows, columns)
df.size                       # Total elements
df.ndim                       # Number of dimensions
df.columns                    # Column names
df.index                      # Row indices
df.dtypes                     # Data types of columns
df.info()                     # Summary information
df.describe()                 # Statistical summary
```

## Data Input/Output

### Reading Data
```python
# CSV files
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', sep=';', header=0, index_col=0)

# Excel files
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# JSON files
df = pd.read_json('file.json')

# SQL databases
df = pd.read_sql('SELECT * FROM table', connection)

# Other formats
df = pd.read_parquet('file.parquet')
df = pd.read_pickle('file.pkl')
```

### Writing Data
```python
# CSV files
df.to_csv('output.csv', index=False)

# Excel files
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# JSON files
df.to_json('output.json', orient='records')

# Other formats
df.to_parquet('output.parquet')
df.to_pickle('output.pkl')
```

## Data Inspection

```python
# First/Last rows
df.head()                     # First 5 rows
df.head(10)                   # First 10 rows
df.tail()                     # Last 5 rows

# Shape and info
df.shape                      # (rows, columns)
df.info()                     # Data types and non-null counts
df.describe()                 # Statistical summary
df.memory_usage()             # Memory usage

# Check for missing values
df.isnull().sum()             # Count of null values per column
df.notnull().sum()            # Count of non-null values
df.isna()                     # Same as isnull()

# Unique values
df['column'].unique()         # Unique values in column
df['column'].nunique()        # Number of unique values
df['column'].value_counts()   # Count of each unique value
```

## Data Selection and Indexing

### Column Selection
```python
df['A']                       # Single column (Series)
df[['A', 'B']]               # Multiple columns (DataFrame)
df.A                          # Column access via attribute (if valid name)
```

### Row Selection
```python
# By position (iloc)
df.iloc[0]                    # First row
df.iloc[0:3]                  # First 3 rows
df.iloc[0:3, 1:3]            # First 3 rows, columns 1-2

# By label (loc)
df.loc[0]                     # Row with index 0
df.loc[0:2]                   # Rows with index 0-2 (inclusive)
df.loc[0:2, 'A':'C']         # Rows 0-2, columns A-C
df.loc[df['A'] > 1]          # Boolean indexing
```

### Boolean Indexing
```python
# Single condition
df[df['A'] > 1]              # Rows where column A > 1
df[df['B'].isin([1, 2, 3])]  # Rows where B is in list

# Multiple conditions
df[(df['A'] > 1) & (df['B'] < 5)]  # AND condition
df[(df['A'] > 1) | (df['B'] < 5)]  # OR condition
df[~(df['A'] > 1)]           # NOT condition (negation)
```

### Query Method
```python
df.query('A > 1')            # Same as df[df['A'] > 1]
df.query('A > 1 and B < 5')  # Multiple conditions
df.query('A in [1, 2, 3]')   # Using 'in' operator
```

## Data Cleaning

### Handling Missing Values
```python
# Remove missing values
df.dropna()                   # Drop rows with any NaN
df.dropna(subset=['A'])       # Drop rows with NaN in column A
df.dropna(axis=1)             # Drop columns with any NaN
df.dropna(thresh=2)           # Keep rows with at least 2 non-NaN values

# Fill missing values
df.fillna(0)                  # Fill all NaN with 0
df.fillna({'A': 0, 'B': 1})   # Fill specific columns
df.fillna(method='ffill')     # Forward fill
df.fillna(method='bfill')     # Backward fill
df.fillna(df.mean())          # Fill with column means
```

### Removing Duplicates
```python
df.duplicated()               # Boolean series indicating duplicates
df.drop_duplicates()          # Remove duplicate rows
df.drop_duplicates(subset=['A'])  # Remove duplicates based on column A
df.drop_duplicates(keep='last')   # Keep last occurrence
```

### Data Type Conversion
```python
df['A'].astype('int')         # Convert column to integer
df['A'].astype('str')         # Convert to string
pd.to_numeric(df['A'])        # Convert to numeric (handles errors)
pd.to_datetime(df['date'])    # Convert to datetime
df['A'].astype('category')    # Convert to categorical
```

## Data Manipulation

### Adding/Removing Columns
```python
# Add columns
df['D'] = df['A'] + df['B']   # New column from calculation
df['E'] = 0                   # New column with constant value
df = df.assign(F=df['A'] * 2) # Assign method

# Remove columns
df.drop('D', axis=1)          # Drop column D
df.drop(['D', 'E'], axis=1)   # Drop multiple columns
del df['D']                   # Delete column in place
```

### Adding/Removing Rows
```python
# Add rows
new_row = pd.Series({'A': 5, 'B': 6})
df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

# Remove rows
df.drop(0)                    # Drop row with index 0
df.drop([0, 1])              # Drop multiple rows
```

### Renaming
```python
# Rename columns
df.rename(columns={'A': 'X', 'B': 'Y'})
df.columns = ['X', 'Y', 'Z']  # Rename all columns

# Rename index
df.rename(index={0: 'first', 1: 'second'})
```

### Sorting
```python
# Sort by values
df.sort_values('A')           # Sort by column A (ascending)
df.sort_values('A', ascending=False)  # Descending
df.sort_values(['A', 'B'])    # Sort by multiple columns

# Sort by index
df.sort_index()               # Sort by row index
df.sort_index(axis=1)         # Sort by column names
```

## Grouping and Aggregation

### GroupBy Operations
```python
# Basic grouping
grouped = df.groupby('A')
grouped.sum()                 # Sum for each group
grouped.mean()                # Mean for each group
grouped.count()               # Count for each group
grouped.size()                # Size of each group

# Multiple grouping columns
df.groupby(['A', 'B']).sum()

# Apply custom function
df.groupby('A').apply(lambda x: x.max() - x.min())

# Aggregation with different functions
df.groupby('A').agg({
    'B': 'sum',
    'C': 'mean',
    'D': ['min', 'max']
})
```

### Pivot Tables
```python
# Create pivot table
pd.pivot_table(df,
               values='D',
               index='A',
               columns='B',
               aggfunc='mean')

# Pivot with multiple values
pd.pivot_table(df,
               values=['C', 'D'],
               index='A',
               columns='B',
               aggfunc='sum')
```

### Cross Tabulation
```python
pd.crosstab(df['A'], df['B'])     # Frequency table
pd.crosstab(df['A'], df['B'], normalize=True)  # Proportions
```

## String Operations

```python
# Accessing string methods
df['text'].str.lower()        # Convert to lowercase
df['text'].str.upper()        # Convert to uppercase
df['text'].str.title()        # Title case
df['text'].str.len()          # Length of strings

# String manipulation
df['text'].str.replace('old', 'new')  # Replace text
df['text'].str.split(' ')     # Split strings
df['text'].str.contains('pattern')    # Boolean mask for pattern
df['text'].str.startswith('prefix')   # Starts with prefix
df['text'].str.endswith('suffix')     # Ends with suffix

# Extract information
df['text'].str.extract(r'(\d+)')      # Extract numbers with regex
df['text'].str.findall(r'\w+')        # Find all words
```

## Date and Time Operations

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter

# Date operations
df['days_ago'] = (pd.Timestamp.now() - df['date']).dt.days
df.set_index('date').resample('M').sum()  # Resample by month
```

## Merging and Joining

### Concatenation
```python
# Concatenate along rows (stack vertically)
pd.concat([df1, df2])
pd.concat([df1, df2], ignore_index=True)

# Concatenate along columns (side by side)
pd.concat([df1, df2], axis=1)
```

### Merging DataFrames
```python
# Inner join (default)
pd.merge(df1, df2, on='key')

# Different join types
pd.merge(df1, df2, on='key', how='left')    # Left join
pd.merge(df1, df2, on='key', how='right')   # Right join
pd.merge(df1, df2, on='key', how='outer')   # Outer join

# Different column names
pd.merge(df1, df2, left_on='key1', right_on='key2')

# Multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])
```

## Statistical Operations

### Basic Statistics
```python
df.mean()                     # Mean of each column
df.median()                   # Median
df.mode()                     # Mode
df.std()                      # Standard deviation
df.var()                      # Variance
df.min()                      # Minimum
df.max()                      # Maximum
df.sum()                      # Sum
df.count()                    # Count of non-null values
```

### Correlation and Covariance
```python
df.corr()                     # Correlation matrix
df['A'].corr(df['B'])         # Correlation between two columns
df.cov()                      # Covariance matrix
```

### Apply Functions
```python
# Apply function to columns
df.apply(np.sum)              # Sum each column
df.apply(lambda x: x.max() - x.min())  # Range of each column

# Apply function to rows
df.apply(np.sum, axis=1)      # Sum each row

# Apply function element-wise
df.applymap(lambda x: x**2)   # Square all elements
```

## Advanced Operations

### Window Functions
```python
# Rolling window
df['A'].rolling(window=3).mean()      # 3-period moving average
df['A'].rolling(window=3).sum()       # 3-period rolling sum
df['A'].rolling(window=3).std()       # 3-period rolling std

# Expanding window
df['A'].expanding().mean()            # Cumulative mean
df['A'].expanding().sum()             # Cumulative sum
```

### Ranking
```python
df['A'].rank()                # Rank values
df['A'].rank(method='dense')  # Dense ranking
df['A'].rank(ascending=False) # Descending rank
```

### Sampling
```python
df.sample(n=5)                # Random sample of 5 rows
df.sample(frac=0.5)           # Random 50% of rows
df.sample(n=5, replace=True)  # Sample with replacement
```

## MultiIndex (Hierarchical Indexing)

```python
# Create MultiIndex
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(4, 2), index=index)

# Access MultiIndex data
df.loc['A']                   # All rows where first level is 'A'
df.loc[('A', 1)]             # Specific multi-index location
df.xs('A', level='first')     # Cross-section

# Stack and unstack
df.stack()                    # Pivot columns to rows
df.unstack()                  # Pivot rows to columns
```

## Performance Tips

### Memory Optimization
```python
# Check memory usage
df.memory_usage(deep=True)

# Optimize data types
df['int_col'] = df['int_col'].astype('int32')     # Use smaller int
df['cat_col'] = df['cat_col'].astype('category')  # Use categorical

# Read large files in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    # Process chunk
    pass
```

### Vectorized Operations
```python
# Use vectorized operations instead of loops
df['C'] = df['A'] + df['B']   # Vectorized (fast)

# Instead of:
# df['C'] = df.apply(lambda row: row['A'] + row['B'], axis=1)  # Slow
```

## Common Patterns

### Filter and Transform
```python
# Filter rows and select columns
result = df[df['A'] > 1][['B', 'C']]

# Chain operations
result = (df
          .query('A > 1')
          .groupby('B')
          .sum()
          .sort_values('C', ascending=False)
          .head(10))
```

### Conditional Operations
```python
# np.where for conditional operations
df['category'] = np.where(df['A'] > 0, 'positive', 'negative')

# Multiple conditions with np.select
conditions = [df['A'] > 0, df['A'] < 0]
choices = ['positive', 'negative']
df['category'] = np.select(conditions, choices, default='zero')
```

### Creating dummy variables
```python
pd.get_dummies(df['category'])        # One-hot encoding
pd.get_dummies(df, columns=['category'])  # Encode specific columns
```

This Pandas cheatsheet covers the essential operations for data analysis and manipulation!