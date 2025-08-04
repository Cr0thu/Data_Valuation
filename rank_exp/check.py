import re
import numpy as np

def calculate_mean_and_variance(file_path):
    # Initialize a list to store the extracted numbers
    numbers = []

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, start=1):
            # Check if the line number is a multiple of 42
            if i % 42 == 0:
                # Use regex to extract the last number in the line
                match = re.search(r"(\d+\.\d+)$", line)
                if match:
                    # Append the extracted number to the list
                    numbers.append(float(match.group(1)))

    # Calculate mean and variance using numpy
    mean = np.mean(numbers)
    variance = np.var(numbers)
    print(numbers)

    return mean, variance

# Specify the path to your file
file_path = 'experiment_log_train100_test100_T2000_K10_penalty10000_fixFalse_randFalse_20250508_135759.txt'

# Call the function and print the results
mean, variance = calculate_mean_and_variance(file_path)
print(f"Mean: {mean}")
print(f"Variance: {variance}")