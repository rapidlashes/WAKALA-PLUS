"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""
import statistics
import csv
import math
from pathlib import Path

RAW_DATA_PATH = Path("Raw") / "agent_deposits.csv"
column_x = 'daily_deposits_kes'


# Step 1: loading our numerical data, the ones we want to work with
def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
    raw_data = []
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        if column not in reader.fieldnames:
            raise ValueError(f"{column} not found in {path}!")
        for row_number, row in enumerate(reader, start=2):
            value = row[column]
            raw_data.append(value)

    return raw_data


raw_data = load_column(
    r'C:\Users\ahmed\Downloads\ASSIGNMENT_3\Data\Raw\agent_deposits.csv', column_x)
print(f"RAW DATA: {raw_data}")
print(f"COUNT OF RAW DATA: {len(raw_data)}")
print()
# NOTE: Loading this raw data, will load 40 instead of 41 values, because our csv data rows start from index 2, (the fieldnames took index 1)
# Thasts why we used the enumerate method.


# Step 2:Now cleaning that numerical data so that we can compute it numerically
def clean(dirty_data: list):
    clean_data = []
    rejects = []
    for x in dirty_data:
        x = x.strip()
        try:
            number = float(x)
            clean_data.append(number)
        except (ValueError, AttributeError):
            rejects.append(x)

    return clean_data, rejects


clean_data, rejects = clean(raw_data)
print(f"CLEANED DATA: {clean_data}")
print(f"COUNT OF CLEAN DATA: {len(clean_data)}")
print()
print(f"REJECTED VALUES: {rejects}")
print(f"COUNT OF REJECTED VALUES:{len(rejects)}")
print()
# NOTE:After cleaning we will remain with 35 values from the 40, dropping '-', '1240O, there is lettr 'O' in our figure we dont want to assume anything so we'll just consider it invalid,
# blanks were dropped, 'n/a' values were dropped


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
    highest_count = 0  # at the moment highest_count is 0
    for x in value_counts.values():  # looping through only the values and not keys in our dictionary value_counts
        if x > highest_count:
            highest_count = x  # playing with indexes, the code recognises the count with the biggest number as highest_count
    if highest_count == 1:
        # an empty dictionary is returned if each value in our lclean_data list appears only once, rendering x(count) = 1
        return {}
    else:
        # if some values appear more than once , return value_counts{key:value} in ascending order of the value
        return sorted(value_counts)


mode = calculate_mode(clean_data)
print(mode)
print()
# the {} in mode means there's no value that has been repeated across the data set


def calculate_variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    mylist = []
    sample = (len(values))-1
    mean = calculate_mean(values)
    for value in values:
        x = value - mean
        y = x**2
        mylist.append(y)
    var = (sum(mylist))/sample
    return var


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
    return q2, q1, q3


q2, q1, q3 = quartiles(clean_data, list1, list2)
print(f"SECOND QUARTILE/MEDIAN : {q2}")
print(f"FIRST QUARTILE: {q1}")
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

    def __init__(self, values: list[float], name: str = "values") -> None:
        self.values = list(values)
        self.name = name
        self.sorted = sorted(values)
        self.rejected = []
        self.removed_outliers = []

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

        return kept

    def summary_exclusive_outliers(self):
        data = self.without_outliers()
        count = len(data)
        mean = sum(data) / count

        # 1. median(should be 18506.72)
        ordered_data = sorted(data)
        floor = count // 2  # 15 is the floor number
        remainder = count % 2  # 1 is the remainder
        if remainder == 1:
            median_no = ordered_data[floor]
        elif remainder == 0:
            median_no = (ordered_data[floor] + ordered_data[floor - 1])/2

        # 2.mode
        empty_dict = {}
        for value in data:
            empty_dict[value] = empty_dict.get(value, 0) + 1
        highest = 0
        for x in empty_dict.values():
            if x > highest:
                highest = x
        if highest == 1:
            print({})
        else:
            print(sorted(empty_dict))

        # 3. variance(sample)
        x = 0
        for v in data:
            var = (v - mean)**2
            x = x + var
        vrc = x/(count-1)

        # standard deviation
        stan_dev = vrc ** 0.5

        # quartile buckets
        first_bucket = ordered_data[:floor]
        second_bucket = ordered_data[-floor:]

        # quartile_1
        flr1 = len(first_bucket)//2
        rem1 = len(first_bucket) % 2
        if rem1 == 1:
            quartile_1 = first_bucket[flr1]
        elif rem1 == 0:
            quartile_1 = (first_bucket[flr1] + first_bucket[flr1-1])/2

        # quartile_3
        flr2 = len(second_bucket)//2
        rem2 = len(second_bucket) % 2
        if rem2 == 1:
            quartile_3 = second_bucket[flr2]
        elif rem2 == 0:
            quartile_3 = (second_bucket[flr2] + second_bucket[flr2-1])/2

        # interquartile range
        interq_range = quartile_3 - quartile_1

        # fences
        lower_f = quartile_1 - (1.5 * interq_range)
        upper_f = quartile_3 + (1.5 * interq_range)

        # outliers
        out = []
        for y in data:
            if y < lower_f or y > upper_f:
                out.append(y)
            else:
                continue

        return {
            'highest value': max(data),
            'lowest value': min(data),
            'count': count,
            'mean': mean,
            'median': median_no,
            'mode': highest,
            'variance': vrc,
            'standard deviation': stan_dev,
            'lower fence': lower_f,
            'first quartile': quartile_1,
            'upper fence': upper_f,
            'third quartile': quartile_3,
            'interquartile range': interq_range,
            'outliers': out,
            'count of outliers': len(out)
        }

    def report(self) -> str:
        all_summary = self.summary()
        minus_outliers_data = self.without_outliers()
        minus_outlier_summary = self.summary_exclusive_outliers()
        lines = []  # the report is built one line at a time
        lines.append('WAKALA PLUS - DAILY DEPOSIT STATISTICS')
        lines.append("=" * 62)
        raw_values = len(raw_data)
        lines.append(f"Original Data: {raw_values} values\n{raw_data}")
        lines.append("")
        cleaned_values = all_summary['count']
        lines.append(f"Cleaned Data: {cleaned_values} values\n{clean_data}")
        lines.append("")
        rejected_values = len(rejects)
        lines.append(f"Rejected values:{rejected_values} values\n{rejects}")
        lines.append("")
        print("\n".join(lines))
        print(f"STATISCTICAL SUMMARY OF THE CLEANED DATA:")
        print(all_summary)
        print()
        print(f"LIST OF CLEANED DATA EXCLUDING OUTLIERS:")
        print(minus_outliers_data)
        print()
        print("STATISTICAL SUMMARY OF CLEANED DATA EXCLUDING OUTLIERS:")
        print(minus_outlier_summary)


stats = StatisticsAnalyzer(clean_data)
print()
print(stats.report())
print()


def compare_with_numpy(values):
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np
    data = np.array(values)

    analyzer = StatisticsAnalyzer(clean_data)
    mine = analyzer.summary()
    # compute each statistic value with Numpy
    # Quartiles
    q1, q3 = np.percentile(data, [25, 75], method='linear')
    iq_range = q3-q1
    lower_fnce = q1-(1.5*iq_range)
    upper_fnce = q3+(1.5*iq_range)

    numpy_stats = {
        'highest value': np.max(data),
        'lowest value': np.min(data),
        'count': np.size(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': 0000,
        'variance': np.var(data),
        'standard deviation': np.std(data),
        'lower fence': lower_fnce,
        'first quartile': q1,
        'upper fence': upper_fnce,
        'third quartile': q3,
        'interquartile range': iq_range,
        'outliers': 0000,
        'count of outliers': np.sum((data < lower_fnce) | (data > upper_fnce))
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


results = compare_with_numpy(clean_data)
print(f"COMPARING CLEAN DATA STATICTICS AGAINST NUMPY STATISTICS:")
print(results)
print()
print('POST COMPARISON REPORT')
print('=' * 62)
print(f"For variance and standard deviation , numpy by default calculates the population variance, while using our analyzer I calculated the Sample variance")


# if __name__ == "__main__":
#     # TODO
#     pass


def save_report(path, report_list):
    x = compare_with_numpy(clean_data)
