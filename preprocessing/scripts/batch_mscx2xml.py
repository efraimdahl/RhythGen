ORI_FOLDER = ""  # Replace with the path to your folder containing mscx (musescore)
DES_FOLDER = ""   # The script will convert the musicxml files and output standard abc notation files to this folder

import os
import math
import random
import subprocess
from tqdm import tqdm
from multiprocessing import Pool


def convert_mscx2xml(file_list):
    cmd = f"musescore3"
    for file in tqdm(file_list):
        filename = os.path.basename(file)
        os.makedirs(DES_FOLDER, exist_ok=True)
        o = os.path.join(DES_FOLDER, filename.rsplit('.', 1)[0] + '.xml')
        try:
            p = subprocess.Popen(cmd + ' "' + file + '" -o "' + o + '" -f', stdout=subprocess.PIPE, shell=True)
            result = p.communicate()
            output = result[0].decode('utf-8')

        except Exception as e:
            with open("logs/midi2xml_error_log.txt", "a", encoding="utf-8") as f:
                f.write(file + ' ' + str(e) + '\n')


if __name__ == '__main__':
    file_list = []
    os.makedirs("logs", exist_ok=True)

    # Traverse the specified folder for XML/MXL files
    for root, dirs, files in os.walk(os.path.abspath(ORI_FOLDER)):
        for file in files[0:10]:
            if file.endswith((".mscx")):
                filename = os.path.join(root, file).replace("\\", "/")
                file_list.append(filename)
    print(file_list)

    # Shuffle and prepare for multiprocessing
    random.shuffle(file_list)
    num_files = len(file_list)
    num_processes = os.cpu_count()
    file_lists = [file_list[i::num_processes] for i in range(num_processes)]

    # Create a pool for processing
    with Pool(processes=num_processes) as pool:
        pool.map(convert_mscx2xml, file_lists)