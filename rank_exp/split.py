def split_file(input_file, lines_per_file=53):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total_lines = len(lines)
    num_files = (total_lines // lines_per_file) + (1 if total_lines % lines_per_file != 0 else 0)

    for i in range(num_files):
        start_line = i * lines_per_file
        end_line = start_line + lines_per_file
        with open(f'analysis/output_file_{i+1}_mnist.txt', 'w', encoding='utf-8') as output_file:
            output_file.writelines(lines[start_line:end_line])

# 使用方法
split_file('mnist_experiment_log_train100_test100_T2000_K20_penalty100_fixFalse_randFalse_20250509_135228.txt', 42)