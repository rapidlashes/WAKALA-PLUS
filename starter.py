"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""
import csv
import math
from pathlib import Path

DATA_PATH = Path("Data") / "agent_deposits.csv"
column_x = 'daily_deposits_kes'


# Step 1: loading our numerical data, the ones we want to work with
def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
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
                # we are using duble brackets because we are appending in our list one tuple containing 2 elements
                rejects.append((row_number, value))

    return clean_data, rejects


clean_data, rejects = load_column(
    DATA_PATH, column_x)
print(f"CLEAN DATA: {clean_data}")
print(f"COUNT OF CLEAN DATA: {len(clean_data)}")
print()
print(f"REJECTED VALUES: {rejects}")
print(f"COUNT OF REJECTED VALUES:{len(rejects)}")
print()


def calculate_mean(values: list[float]) -> float:
    total = 0
    for value in values:
        total = total + value
    mean = total/len(values)
    return mean


mean = calculate_mean(clean_data)
print(f"MEAN: {mean}")
print()


def calculate_median(values: list[float]) -> float:
    ordered = sorted(values)
    x = len(ordered)
    y = x % 2  # if 1 then its odd, if 0 then its even
    z = x // 2
    if y == 1:
        median_number = ordered[z]
    elif y == 0:
        median_number = (ordered[z] + ordered[z-1])/2
    return median_number


median = calculate_median(clean_data)
print(f"MEDIAN: {median}")
print()


def calculate_mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""
    value_counts = {}
    for value in values:
        # for every value it loops , adds a count of 1, if it finds it again, it adds on the count
        value_counts[value] = value_counts.get(value, 0) + 1
    highest_count = 0
    for x in value_counts.values():  # looping through only the values and not keys in our dictionary value_counts
        if x > highest_count:
            highest_count = x  # playing with indexes, the code recognises the count with the biggest number as highest_count
    if highest_count == 1:
        return []
    modes = []
    for value, x in value_counts.items():
        if x == highest_count:
            # only shows the value, doesnt show its count, best believe that value  has the highest count
            modes.append[value]


mode = calculate_mode(clean_data)
print(mode)
print()
# the [] in mode means there's no value that has been repeated across the data set


def calculate_variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    n = len(values)
    if n == 0:
        raise ValueError("Cannot calculate variance of an empty list.")
    if sample and n < 2:
        # if n= 1, n-1 = 0, cant divide by zero, we will get zerodivision error, so im basically catching that error
        raise ValueError('Sample Variance requires atleast 2 values')
    mean = calculate_mean(values)
    squared_difference = []
    for value in values:
        difference = value - mean
        squared = difference ** 2
        squared_difference.append(squared)
        denominator = (n-1) if sample else n

    variance = (sum(squared_difference))/denominator
    return variance


variance = calculate_variance(clean_data)
print(variance)
print()


def std_dev(values: list[float], sample: bool = True) -> float:
    var = calculate_variance(values)
    std = var ** 0.5
    return std


standard_deviation = std_dev(clean_data)
print(standard_deviation)


def quartile_brackets(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3). Document which quartile method you chose."""
    ordered = sorted(values)
    length = len(ordered)
    index = length // 2
    list1 = ordered[:index]
    list2 = ordered[-index:]

    return ordered, list1, list2


ordered, list1, list2 = quartile_brackets(clean_data)
print(f"ORDERED DATA: {ordered}")
print()
print(f"FIRST BRACKET: {list1}")
print()
print(f"SECOND BRACKET: {list2}")
print()


def quartiles(values, first_list, second_list):
    ordered = sorted(values)
    q2 = calculate_median(ordered)
    q1 = calculate_median(first_list)
    q3 = calculate_median(second_list)
    return q1, q2, q3


q1, q2, q3 = quartiles(clean_data, list1, list2)
print(f"FIRST QUARTILE: {q1}")
print(f"SECOND QUARTILE/MEDIAN : {q2}")
print(f"THIRD QUARTILE: {q3}")
print()


def iqr_outliers(values: list[float], lower_quartile, upper_quartile, k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    outliers = []
    ordered = sorted(values)
    iqr = upper_quartile - lower_quartile
    lower_bound = lower_quartile - (k * iqr)
    upper_bound = upper_quartile + (k * iqr)
    for value in ordered:
        if value < lower_bound:
            outliers.append(value)
        elif value > upper_bound:
            outliers.append(value)
    return iqr, outliers, lower_bound, upper_bound


iqr, outliers, lower_bound, upper_bound = iqr_outliers(clean_data, q1, q3)
print(f"INTERQUARTILE RANGE: {iqr}")
print(f"OUTLIERS: {outliers}")
print()


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    @classmethod
    def from_csv(cls, path, column):
        clean_data, rejects = load_column(path, column)
        analyzer1 = cls(clean_data, name=column)
        analyzer1.rejected = rejects
        return analyzer1

    def __init__(self, values: list[float], name: str = "values") -> None:
        self.values = list(values)
        self.name = name
        self.sorted = sorted(values)
        self.rejected = []
        self.the_removed_outliers = []

    def summary(self) -> dict:
        return {
            'highest value': max(clean_data),
            'lowest value': min(clean_data),
            'count': len(clean_data),
            'mean': mean,
            'median': median,
            'mode': mode,
            'variance': variance,
            'standard deviation': standard_deviation,
            'lower fence': lower_bound,
            'first quartile': q1,
            'upper fence': upper_bound,
            'third quartile': q3,
            'interquartile range': iqr,
            'outliers': outliers,
            'count of outliers': len(outliers)
        }

    def without_outliers(self) -> "StatisticsAnalyzer":
        info = self.summary()
        kept = []
        x = info['lower fence']
        y = info['lower fence']
        for value in self.values:
            if value < info['lower fence'] or value > info['upper fence']:
                continue
            else:
                kept.append(value)

        # create a new statisticsAnalyzer object, which in it we will now pass our list excluding outliers
        analyzer2 = StatisticsAnalyzer(kept, name=self.name)

        # kept has the evenly distributed values, outliers excluded
        # new_analyzer has now the excluded outliers
        analyzer2.rejected = self.rejected.copy()
        analyzer2.the_removed_outliers = [
            value for value in self.values
            if value < info['lower fence'] or value > info['upper fence']
        ]

        # returning a whole object instead of just the list excluding outliers, kept[]
        return analyzer2

    def summary_exclusive_outliers(self):
        analyzer2_data = self.without_outliers()  # calling out the above function , but it returns a whole object, and in that object we had passed our excluded outlier data, so basically if I want to find the summary of that excluded outlier data, I will just use .summary() method on that object
        return analyzer2_data.summary()

    def report(self) -> str:
        # all_summary is an object, it returns summary of the statistics we did earlier on for our clean data inform of a dictionary
        all_summary = self.summary()
        minus_outliers_data = self.without_outliers()
        minus_outlier_summary = self.summary_exclusive_outliers()

        # the report is built one line at a time
        lines = []
        lines.append('WAKALA PLUS - DAILY DEPOSIT STATISTICS')
        lines.append("=" * 62)

        # since all_summary is in for m of a dict, we want to display the value of key 'count' in that dict
        cleaned_values = all_summary['count']
        lines.append(f"Valid Values: {cleaned_values} values\n{clean_data}")
        lines.append("")

        rejected_values = len(rejects)
        lines.append(f"Rejected values:{rejected_values} values\n{rejects}")
        lines.append("")

        # explaining the logic behind the validity
        lines.append(f" '' - blank value")
        lines.append(f" (-) - an invalid string character ")
        lines.append(f" 'n/a' - a value is missing")
        lines.append(f" '12,40O' - a letter at the end of the float")
        lines.append(f" '' - another blank value")
        lines.append("")

        # unpacking our dictionary of general summary
        lines.append(f"STATISCTICAL SUMMARY OF THE CLEANED DATA:")
        for key, value in all_summary.items():
            lines.append(f"{key} : {value}")

        lines.append("")
        lines.append(f"LIST OF CLEANED DATA EXCLUDING OUTLIERS:")

        # I am using .values because we passed kept[] as our values in the new object we returned from the without_outliers() method. Remember we defined our StatistisAnalyzer claass like this def __init__(self, values,name) and self.values = values
        lines.append(f"{minus_outliers_data.values}")
        lines.append("")
        lines.append(f"THE REMOVED OUTLIERS:")
        lines.append(f"{minus_outliers_data.the_removed_outliers}")
        lines.append(f"{minus_outliers_data}")
        lines.append("")

        # unpacking our dictionary of minus outliers summary
        lines.append(
            f"STATISTICAL SUMMARY OF CLEANED DATA EXCLUDING OUTLIERS:")
        for k, v in minus_outlier_summary.items():
            lines.append(f"{k} : {v}")
        lines.append("")
        print("\n".join(lines))

        return lines

    def save_report(self, filepath):
        repo = self.report()
        with open(filepath, 'w')as file:
            for v in repo:
                file.write(f"{v}\n")


stats1 = StatisticsAnalyzer.from_csv(DATA_PATH, column_x)
report_path = Path(__file__).resolve().parent / 'report.txt'
print()
print(stats1.report())
print()
print(stats1.save_report(report_path))
print()


def compare_with_numpy(values):
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np
    data = np.array(values)

    # instead of using 'stats' which is the object that I used previously, I created another one 'analayzer'
    analyzer = StatisticsAnalyzer(clean_data)
    mine = analyzer.summary()

    # compute each statistic value with Numpy
    # Quartiles and outliers
    q1, q3 = np.percentile(data, [25, 75], method='linear')
    iq_range = q3-q1
    lower_fnce = q1-(1.5*iq_range)
    upper_fnce = q3+(1.5*iq_range)
    numpy_outliers = [(data < lower_fnce) | (data > upper_fnce)]

    # we cant calculate mode directly using numpy instead:
    unique_values, counts = np.unique(data, return_counts=True)
    highest_count = np.max(counts)
    numpy_mode = (
        unique_values[counts == highest_count].tolist() if highest_count > 1 else [])

    numpy_stats = {
        'highest value': np.max(data),
        'lowest value': np.min(data),
        'count': np.size(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': numpy_mode,

        # ddof = 1 will calculate the sample variance(n-1), but `ddof = 0' will calculate population variance (n)
        'variance': np.var(data, ddof=1),
        'standard deviation': np.std(data, ddof=1),
        'lower fence': lower_fnce,
        'first quartile': q1,
        'upper fence': upper_fnce,
        'third quartile': q3,
        'interquartile range': iq_range,
        'outliers': numpy_outliers,
        'count of outliers': np.sum(numpy_outliers)
    }

    print(f"STATISTICAL SUMMARY USING NUMPY")
    print(numpy_stats)
    print()
    results = {}
    for key, value in mine.items():
        numpy_value = numpy_stats[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            results[key] = ('match' if math.isclose(
                value, numpy_value, rel_tol=1e-9, abs_tol=1e-9) else 'mismatch')
        else:
            results[key] = (
                'match' if value == numpy_value else 'mismatch'
            )
    return results


def save_numpy_report(filepath):
    numpy_report_lines = []
    numpy_report_lines.append(
        f"COMPARING OUR INTITAL STATISTICS AGAINST NUMPY STATISTICS:")
    numpy_report_lines.append('=' * 62)
    results = compare_with_numpy(clean_data)
    for key, value in results.items():
        numpy_report_lines.append(f"{key} : {value}")
    numpy_report_lines.append("")

    with open(filepath, 'w')as file:
        for line in numpy_report_lines:
            file.write(f"{line}\n")


filepath = Path(__file__).resolve().parent / 'numpy_report.txt'
x = save_numpy_report(filepath)

print(x)

# if __name__ == "__main__":
#     # TODO
#     pass
