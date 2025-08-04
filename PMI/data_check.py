import pandas as pd
import sys

def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print("Good Copy:")
    for i in range(0,len(lines),10):
        main_info = lines[i]
        main_info = main_info.strip('\n')
        train_sizes = main_info.split(";")[0].split(":")[1].strip().split(", ")
        train_sizes = [int(x) for x in train_sizes]
        test_sizes = main_info.split(";")[1].split(":")[1].strip().split(", ")
        test_sizes = [int(x) for x in test_sizes]
        penalty_info = lines[i+1].strip('\n').split("&")[0].strip()

        for j in range(1,8,3):
            pmi_data = lines[i+j].strip('\n').strip('\\').split("&")
            loss_data = lines[i+j+1].strip('\n').strip('\\').split("&")
            acc_data = lines[i+j+2].strip('\n').strip('\\').split("&")
            error_info = pmi_data[1].strip().split("\\")[0]
            if (float(pmi_data[4].strip()) < 0):
                print("Train Sizes:", train_sizes, "Test Sizes:", test_sizes)
                print("Penalty:", penalty_info, "Error Ratio:",error_info)
    

    print("Good Delete:")
    for i in range(0,len(lines),10):
        main_info = lines[i]
        main_info = main_info.strip('\n')
        train_sizes = main_info.split(";")[0].split(":")[1].strip().split(", ")
        train_sizes = [int(x) for x in train_sizes]
        test_sizes = main_info.split(";")[1].split(":")[1].strip().split(", ")
        test_sizes = [int(x) for x in test_sizes]
        penalty_info = lines[i+1].strip('\n').split("&")[0].strip()

        for j in range(1,8,3):
            pmi_data = lines[i+j].strip('\n').strip('\\').split("&")
            loss_data = lines[i+j+1].strip('\n').strip('\\').split("&")
            acc_data = lines[i+j+2].strip('\n').strip('\\').split("&")
            error_info = pmi_data[1].strip().split("\\")[0]
            if float(pmi_data[5].strip()) < 0 and float(loss_data[5].strip()) < 0 and float(acc_data[5].strip()) > 0.1 and float(pmi_data[7].strip()) < 0 and float(loss_data[7].strip()) < 0 and float(acc_data[7].strip()) > 0.06:
                print("Train Sizes:", train_sizes, "Test Sizes:", test_sizes)
                print("Penalty:", penalty_info, "Error Ratio:",error_info)

    print("Good Denoise:")
    for i in range(0,len(lines),10):
        main_info = lines[i]
        main_info = main_info.strip('\n')
        train_sizes = main_info.split(";")[0].split(":")[1].strip().split(", ")
        train_sizes = [int(x) for x in train_sizes]
        test_sizes = main_info.split(";")[1].split(":")[1].strip().split(", ")
        test_sizes = [int(x) for x in test_sizes]
        penalty_info = lines[i+1].strip('\n').split("&")[0].strip()

        for j in range(1,8,3):
            pmi_data = lines[i+j].strip('\n').strip('\\').split("&")
            loss_data = lines[i+j+1].strip('\n').strip('\\').split("&")
            acc_data = lines[i+j+2].strip('\n').strip('\\').split("&")
            error_info = pmi_data[1].strip().split("\\")[0]
            if float(pmi_data[8].strip()) > 12 and float(loss_data[8].strip()) < 0 and float(acc_data[8].strip()) > 0.05:
                print("Train Sizes:", train_sizes, "Test Sizes:", test_sizes)
                print("Penalty:", penalty_info, "Error Ratio:",error_info)

file_path = "output.txt"

parse_file(file_path)