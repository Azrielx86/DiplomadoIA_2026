from typing import List

grades: List[float] = [
    8.3,
    8.2,
    9.17,
    9.3,
    8.2,
    8.96,
    8.2,
    8.46,
    8.6,
    8.51,
    8.87,
    10.0,
    8.5,
    10.0,
    8.32,
    8.6,
    9.03,
    7.7,
    8.3,
    9.1,
    8.2,
    8.7,
    8.9,
    8.9,
    8.01,
    4.9,
    8.6,
    9.1,
    8.83,
    9.2,
    9.1,
    9.4,
    9.06,
    8.2,
    8.2
]

max_grade: float = 0
min_grade: float = 10.0
average: float = 0
approved_count: int = 0

tmp_sum: float = 0
for grade in grades:
    max_grade = max(grade, max_grade)
    min_grade = min(grade, min_grade)
    tmp_sum += grade

    if grade > 6.0:
        approved_count += 1

average = tmp_sum / len(grades)

print("Total grades:", len(grades))
print(f"Max grade: {max_grade}")
print(f"Min grade: {min_grade}")
print(f"Average: {average:.2f}")
print(f"Approved count: {approved_count}")
