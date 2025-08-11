ORI_FOLDER = "../../data/outputs/Lieder/LB"
DES_FOLDER = "../../data/example/LB/xml"   # The script will convert the abc files and output musicxml files to this folder

import os
import math
import random
import subprocess
from tqdm import tqdm
from multiprocessing import Pool

def convert_abc2xml(file_list):
    cmd = 'python abc2xml.py '
    for file in tqdm(file_list):
        filename = file.split('/')[-1]  # Extract file name
        os.makedirs(DES_FOLDER, exist_ok=True)

        try:
            p = subprocess.Popen(cmd + '"' + file + '"', stdout=subprocess.PIPE, shell=True)
            result = p.communicate()
            output = result[0].decode('utf-8')

            if output == '':
                with open("preprocessing/logs/abc2xml_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(file + '\n')
                continue
            else:
                output_path = f"{DES_FOLDER}/" + ".".join(filename.split(".")[:-1]) + ".xml"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            with open("preprocessing/logs/abc2xml_error_log.txt", "a", encoding="utf-8") as f:
                f.write(file + ' ' + str(e) + '\n')
            pass

if __name__ == '__main__':
    file_list = []
    os.makedirs("logs", exist_ok=True)
    # Traverse the specified folder for ABC files
    for root, dirs, files in os.walk(ORI_FOLDER):
        print(files)
        for file in files:
            if not file.endswith(".abc"):
                continue
            filename = os.path.join(root, file).replace("\\", "/")
            file_list.append(filename)

    # Prepare for multiprocessing
    file_lists = []
    random.shuffle(file_list)
    for i in range(os.cpu_count()):
        start_idx = int(math.floor(i * len(file_list) / os.cpu_count()))
        end_idx = int(math.floor((i + 1) * len(file_list) / os.cpu_count()))
        file_lists.append(file_list[start_idx:end_idx])

    pool = Pool(processes=os.cpu_count())
    pool.map(convert_abc2xml, file_lists)