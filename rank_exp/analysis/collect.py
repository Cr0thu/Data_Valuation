import os

def extract_scores_from_files(folder_path, output_file):
    LMI_scores = []
    PMI_scores = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.startswith("output_file_") and filename.endswith("mnist.txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
                # 提取LMI_score和PMI_score
                LMI_score = [float(lines[4*i].strip().split()[-1]) for i in range(1, 11)]
                PMI_score = []
                for line in lines:
                    if "Average score for rho" in line:
                        score = float(line.split(':')[1].split('(')[0].strip())
                        PMI_score.append(score)
                
                LMI_scores.append(LMI_score)
                PMI_scores.append(PMI_score)

    # 将结果写入输出文件
    with open(output_file, 'w') as out_file:
        out_file.write("LMI_scores = [\n")
        for score in LMI_scores:
            out_file.write('    [' + ', '.join(map(str, score)) + '],\n')
        out_file.write("]\n\n")
        
        out_file.write("PMI_scores = [\n")
        for score in PMI_scores:
            out_file.write('    [' + ', '.join(map(str, score)) + '],\n')
        out_file.write("]\n")

# 使用示例
folder_path = '../analysis'
output_file = 'scores_output_mnist.txt'
extract_scores_from_files(folder_path, output_file)