import numpy as np
from datetime import datetime

# Read birthdates from the file
with open('date.txt', 'r') as file:
    birthdates = file.readlines()

# Calculate ages
current_date = datetime.now()
ages = []

for birthdate in birthdates:
    birthdate = birthdate.strip()
    birth_date = datetime.strptime(birthdate, '%Y%m%d')
    age = (current_date - birth_date).days / 365.25
    ages.append(age)

# Calculate average age and standard deviation
average_age = np.mean(ages)
std_dev_age = np.std(ages)

print(f"Average Age: {average_age:.2f} years")
print(f"Standard Deviation of Age: {std_dev_age:.2f} years")
