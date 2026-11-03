## WAKALA PLUS 
Wakala Plus runs a network of M-Pesa agents around Nairobi, Thika, Machakos and Nakuru.
Its risk team watches daily deposit totals per agent: unusually high or low days can mean fraud, a float shortage or a data-entry error.

### Problem Statement
WAKALA PLUS wants a small, dependable statistics toolkit they understand line by line, not a black box. 
Every calculation should be implemented in Python first; 
NumPy is permitted only afterwards, to verify that your numbers agree. 

#### Learning objectives
1. Implement mean, median, mode, variance and standard deviation from first principles
2. Clean a numeric column by rejecting missing and invalid values with a recorded reason
3. Detect outliers with the interquartile range (IQR) method
4. Package the calculations in a reusable StatisticsAnalyzer class
5. Verify your results against NumPy and explain any differences
6. Write unit tests that pin down edge cases

#### Dependencies
```python
import statistics 
import csv
import math
from pathlib import Path
```
We use the module `Path`  to handle directories, instead of hard coding our local directory into the code. That way incase somebody else runs the code from their machine, they can just define their own directory.
The `csv` module is used to read our csv file, extracting only the data we need from the file rather than the whole file.

#### STEPS I APROACHED
##### Step 1 - missing aand invalid values
`load_column(path, column)` reads the CSV and  returns two lists: the clean values and rejected values.
We used enumerate since when reading our csv the values column started from row no. 2 in our csv file.
```python
def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    clean_data = []
    rejects = []
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        if column not in reader.fieldnames:
            raise ValueError(f"{column} not found in {path}!")
        for row_number, row in enumerate(reader, start=2):
            value = row[column]
            try:
                number = float(value.strip())
                clean_data.append(number)
            except (ValueError, AttributeError):
                rejects.append((row_number, value))

    return clean_data, rejects

```

**My rule:** a value is accepted only if I can be sure what number it means. That means plain digits (20715.67), or digits with correctly placed thousands commas (20,715.67), and not negative.
Nothing is guessed or repaired. Each rejection is printed in the `report.txt` file with its reason.

**NOTE:** One of the values rejected was `12,40O` .This is probably a typo for `12400` but I rejected it anyway.
This data is used to spot errors and fraud, so I would rather list the row and let a person check the agent's records than quietly invent a figure.

##### Step 2: Mean, mode, median(clean values)
1. **Mean:** add everything up with a loop, divide by how many values there are.
2. **Median:** sort the values. With an odd count, take the middle one. With an even count, average the two middle ones.
3. **Mode:** Count how often each value appears and return a list
4. If each value appears once , return an empty list since there is no highest count but if there is a highest count , return a list of that value specifically.

##### Step 3:  Variance , Standard Deviation
For `variance` I first find the mean, then add up each value's squared distance from it, then divide.
The `standard deviation` is the square root of the variance.
Our function says by default the program calculates for the sample variance (n-1), but someone else running this code on a different data set might decide to check the boolean condition for the sample to be `False`, and the code will automatically calculate Population variance.

** Why we used Sample variance??** 
These 35 days are a slice of something that keeps going: that is the reason why we settled for sample variance.
Agents deposit money every day. What the risk team wants to know is how much deposits normally vary, not just how much these particular 35 days varied.
 On this data the difference between the 2 variance methods is small (18,111 vs 17,851, about 1.5%), 
so the choice  of what Variance method to use matters less than the habit of picking the right one. 
The Population Variance (n) would be correct only if these 35 days were all the days anyone cared about, but that's not the case.

##### Step 4: Finding outliers with the IQR method
1. Sort the values and find Q1 (the 25% mark) and Q3 (the 75% mark).
2. The IQR is Q3 − Q1, the spread of the middle half of the days.
3. The fences are Q1 − 1.5 × IQR (lower) and Q3 + 1.5 × IQR (upper). Anything outside them is an `outlier`. A value exactly on a fence is not one.

`def quartile_brackets(values)` : returns two lists: `list1` (the first 17 values) and `list2` (the last 17 values). The middle value not touched is `Q2` which stands for the `median`.
Now the get `Q1` we found the median of the first list, while `Q3` the second list.

##### Step 5: The StatisticsAnalyzer class
This part has several methods:
1. `.summary()` - This method returns a dictionary with each statistic against its value.eg `{count:35, mean:22400,...}` for all the statistics we had done earlier(global values) on our clean data.
2.  `.without_outliers()` - It gives back a list `kept` of the values excluding the outliers but instead of returning the list, It returns an object defined from the class StatisticsAnalyzer. Inside the object is now the list `kept` passed as a parameter.
3.  `.summary_exclusive_outliers` - It returns a dictionary of the same order of statistics as the first one except it uses the `kept` list from method 2
    **NOTE:**, the second statistics has 0 values for outliers. Our data appears nearly perfect with its values evenly distributed.
4. `.report()` - Returns a report from the above methods;
  - A list of our raw values
  - A list of our cleaned values
  - A list of the rejected values(and an explanation justifying their rejections) 
  - Initial summary of our clean data
  - The summary of  excluding outliers data
  - The list of excluding outliers data
  - A list of the excluded outliers themselves
                                                                
5. `save_report` - Saves the data from our `.report()` which is in form of a list, into a `.txt` file.
We also used a classmethod for loading our csv data in our class.
##### Step 6: Verification of the Statistics
`compare_with_numpy()` recalculates all the statistics parameters above using numpy, compares the results with the values from our initial summary and  returns a dictionary , dict[str, str], in this format; `{count : match, mean : match, variance : mismatch,.....}`. The `match` means the values produced by numpy are the same or close to those of the initial summary.
We also had to import the `math` module to implement this function.
```python
import math
```
`save_numpy_report(filepath)` -  Saved this comparison dictionary into another `.txt` file.

#### WORTH NOTING...
All the executable code is put inside the `if __name__ == "__main__":` guard.
Why are we doing this....
Consider a situation where you are running a python file directly
```powershell
python starter.py
```
python sees `if __name__ == '__main__'` asuumes the condition is `True` since you are running it directly and executes the codes beneath it. An alternative situation is you are importing the class from starter.py to another .py file 
```python
from starter import StatisticAnalyzer
```
everything loads including the class definitions and functions but the code inside the guard doesnt execute, so one can reuse the classes automatically without loading the csv's and generating the reports.
1. 


