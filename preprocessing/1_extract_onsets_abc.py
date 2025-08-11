
import os
from .abc_rhythm import ABCRhythmTool
import multiprocessing
import pickle
import tqdm
import json
import numpy as np
import math
from functools import partial
from .data_config import *

ORI_FOLDER = "data/example/LB/abc" # Replace with the path to your folder containing standard ABC notation files
ONSET_FILE_LOCATION="data/labels/" # Output target information here
ONSET_FILE_NAME = "LB_onsets.json" # Output onset information here it will be saved at ONSET_FILE_LOCATION/ONSET_FILE_NAME

INTERLEAVED = False # Set this to match the representation of the input, True if the ABC is interleaved, False otherwise.

MANUAL_BOUNDARIES = [None,None] #Set to none if you want to use statistical labelling, set to a list of boundaries otherwise i.e [3,6,8,12,18,24,48] Make sure this matches your number of classes.

TARGET_ONSET_FILE = None #This is where quantile boundaries are calculated from. Set to None to calculate boundaries from current dataset. 

FEATURES = ["densities","syncopation"]

NUM_QUANTILES = N_CLASSES - 1

os.makedirs(ONSET_FILE_LOCATION, exist_ok=True)

tools = ABCRhythmTool(min_note_val=1/GRID_RESOLUTION)

def extract_onsets(abc_path, rotated=INTERLEAVED):
    """
    Extracts MIDI instruments from a given MIDI file and maps them to instrument names and classes.

    Args:
        midi_file_path (str): Path to the MIDI file.
    Returns:
        dict: A dictionary mapping MIDI channels to their instrument class and name.
    """
    try:

        onset_dic = tools.extract_unique_onsets(abc_path, rotated=rotated)
        

        return {
                "file_path":abc_path,
                "onsets":onset_dic["onsets"],
                "densities":onset_dic["densities"],
                #"distances":onset_dic["distances"], 
                "meter": onset_dic["meter"],
                "spectral_arranged":onset_dic["spectral_arranged"],
                "metric_arranged":onset_dic["metric_arranged"],
                "spectral_arranged_treble":onset_dic["spectral_arranged_treble"],
                "metric_arranged_treble":onset_dic["metric_arranged_treble"],

                "syncopation":onset_dic["syncopation"]
                }
    except Exception as e:
        print(f"Error processing {abc_path}: {e}")
        return None


def extract_with_timeout(pool, func, arg, timeout):
    result = pool.apply_async(func, (arg,))
    try:
        return result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        print(f"Timeout on: {arg}")
        return None

def process_abc_directory(directory, output_pickle):
    """
    Processes all MIDI files in a directory to extract instrument information
    and saves the results to a pickle file.

    Args:
        directory (str): Path to the directory containing MIDI files.
        output_pickle (str): Path to save the extracted instrument data as a pickle file.
        midimap (pd.DataFrame): DataFrame containing MIDI program mappings.
    """
    
    # Get all MIDI files in the directory
    abc_files = [
    os.path.join(root, f) 
    for root, _, files in os.walk(directory) 
    for f in files if f.endswith('.abc')
    ]

    #abc_files = abc_files[0:2]

    if not abc_files:
        print("No ABC files found in the directory.")
        return

    #abc_files=abc_files[0:10]
    num_processes = os.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for f in tqdm.tqdm(abc_files, desc="Extracting Onsets"):
            res = extract_with_timeout(pool, extract_onsets, f, timeout=60)  # timeout in seconds
            results.append(res)


    # Filter out empty results
    all_onsets = [result for result in results if result]
    print(f"Onset extraction failed for {len(results)-len(all_onsets)} out of {len(results)} files")
    return all_onsets


def label_abc_file(abc_path,quantile_edges,data=None,idx=None,feature="densities"):
    
    onsets,feature_vals = data[idx]["onsets"],data[idx][feature]
    
    
    labels = []
    for feature in feature_vals:
        if(feature==0):
            labels.append(0)
        else:
            feature = max(quantile_edges[0], min(feature, quantile_edges[-1]))  # Clamp density to the range of quantile edges
            for i in range(NUM_QUANTILES):
                if quantile_edges[i] <= feature <= quantile_edges[i + 1]:
                    labels.append(i + 1)  # Convert to 1-based index
                    break
    if(idx==None or data==None):
        data[idx]["label"] = labels
    
    return labels




def process_onset(i, quantile_collection):
    x = onsets[i]
    file_path = x["file_path"]
    package_dict = {"densities": x["densities"], 
                    #"distances": x["distances"], 
                    "onsets":x["onsets"], 
                    "metric_arranged":x["metric_arranged"],
                    "spectral_arranged":x["spectral_arranged"],
                    "metric_arranged_treble":x["metric_arranged_treble"],
                    "spectral_arranged_treble":x["spectral_arranged_treble"],
                    "syncopation":x["syncopation"],
                    "meter":x["meter"]}

    for feature in FEATURES:
        feature_label = label_abc_file(file_path, quantile_collection[feature], data=onsets, idx=i, feature=feature)
        package_dict[f"{feature}_labels"] = feature_label

    return file_path, package_dict


if __name__=="__main__":
    if(TARGET_ONSET_FILE):
        with open(TARGET_ONSET_FILE, "r") as file:
            nonsets = json.load(file)
    
    onsets = process_abc_directory(ORI_FOLDER,ONSET_FILE_LOCATION)

    print("Successfully extracted onsets, now labelling data")
    
    feature_collection = {}

    if(TARGET_ONSET_FILE):
        with open(TARGET_ONSET_FILE, "r") as file:
            nonsets = json.load(file)
            for feature in FEATURES:
                featurelis = []
                for key,value in nonsets.items():
                    featurelis += value[feature]   
                feature_collection.update({feature:featurelis})
    else:
        for feature in FEATURES:
            featurelis = []
            for entry in onsets:
                featurelis += entry[feature]
            feature_collection.update({feature:featurelis})

    onset_data = {}
    quantile_collection = {}
    #print(set(feature_collection["distances"]))
    for feature_index, feature in enumerate(FEATURES):
        if(MANUAL_BOUNDARIES[feature_index]!=None):
            quantile_edges = MANUAL_BOUNDARIES[feature_index]
        else:
            quantile_edges = np.quantile(feature_collection[feature], np.linspace(0, 1, NUM_QUANTILES + 1))
        quantile_collection[feature]=quantile_edges
    print("Quantiles:",quantile_collection)
    process_onset_with_features = partial(process_onset, quantile_collection=quantile_collection)

    with multiprocessing.Pool() as pool:
        results = list(tqdm.tqdm(pool.imap(process_onset_with_features, range(len(onsets))), total=len(onsets)))

    onset_data = {file_path: package_dict for file_path, package_dict in results}

    json_file = json.dumps(onset_data)
    with open(f"{ONSET_FILE_LOCATION}/{ONSET_FILE_NAME}", "w") as file:
        file.write(json_file)
        

