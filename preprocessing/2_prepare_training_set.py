
ORI_FOLDER = "data/example/LB/abc"  # Replace with the path to your folder containing standard ABC notation files
INTERLEAVED_FOLDER = "data/example/LB_training/interleaved"   # Output interleaved ABC notation files to this folder
AUGMENTED_FOLDER = 'data/example/LB_training/augmented'   # Output key-augmented and rest-omitted ABC notation files to this folder

ONSET_DATA = 'data/labels/LB_onsets.json' #Point this to your extracted onset file.

FEATURE_TYPE = "arranged" #Set to bar for one feature per bar. Set to onset for one features per onset, Set to arranged for one feature per onset already arranged by

TARGET_FEATURE = "spectral_arranged"

PATH_REPLACEMENT_TERM = "abc_test" #"abc_test" #if onset key differes from file location. (default is abc)

import copy
import os
import re
import json
import shutil
import random
from .data_config import *
from tqdm import tqdm
from abctoolkit.utils import (
    remove_information_field, 
    remove_bar_no_annotations, 
    Quote_re, 
    Barlines,
    extract_metadata_and_parts, 
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict)
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.transpose import Key2index, transpose_an_abc_text

os.makedirs(INTERLEAVED_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

if(USE_AUGMENTED):
    for key in Key2index.keys():
        key_folder = os.path.join(AUGMENTED_FOLDER, key)
        os.makedirs(key_folder, exist_ok=True)

try:
    # Opening JSON file
    with open(ONSET_DATA, 'r') as openfile:
        # """Reading from json file
        onset_data = json.load(openfile)
except:
    onset_data= None
    raise ValueError(f"No onset data available at {openfile}")
    print(f"No onset data available at {ONSET_DATA}, proceeding without onset labels")



def split_and_append_ima(abc_lines,abc_path, meter):
    if(ONSET_DATA==None):
        return abc_lines
    
    f_lines = []
    header = []
    voice_lines = []
    in_header = True

    for line in abc_lines:
        if in_header and not line.startswith("[V:"):
            header.append(line)
        else:
            in_header = False
            if line.startswith("[V:"):
                voice_lines.append(line)

    
    beats, beat_unit = map(int, meter.split("/"))  # e.g., 4/4 -> beats=4, beat_unit=4

    if(beats/beat_unit)>MAX_RATIO:
        raise ValueError(f"Meter {meter}, exceed allowed ratio of {MAX_RATIO}",abc_path)
    # Total duration of the bar in units of 1/max_denom notes
    bar_duration = int((beats / beat_unit) * GRID_RESOLUTION)
    
    rpath = abc_path.replace(PATH_REPLACEMENT_TERM,"abc")

    labels = copy.copy(onset_data[rpath][TARGET_FEATURE])

    label_lines = []
    curdex = 0
    while curdex<len(voice_lines):
        target = min(curdex+MAX_BAR_NUM,len(voice_lines))
        if(MIN_BAR_NUM and (target-curdex)<MIN_BAR_NUM):
            break

        bar_control_lines = []
        for current_bar_index in range(curdex,target):
            bar_target_index = min((current_bar_index+1)*bar_duration,len(labels))

            bar_onsets = labels[current_bar_index*bar_duration:bar_target_index]
            bar_paddings = [0] * (int(MAX_RATIO*GRID_RESOLUTION)-len(bar_onsets))   
            #print(len(bar_onsets),len(bar_paddings), current_bar_index, len(labels), len(voice_lines))
            total_bar_onsets = bar_onsets+bar_paddings
            assert len(total_bar_onsets)==int(GRID_RESOLUTION*MAX_RATIO), f"bar features are not grid aligned {len(total_bar_onsets)} vs {GRID_RESOLUTION*MAX_RATIO}"
            line = [f"%{CONTROL_TERM} {' '.join([str(d) for d in total_bar_onsets])}\n"]
            
            bar_control_lines+=line
        musicLines = bar_control_lines+header+voice_lines[curdex:target]
        f_lines.append(musicLines)
        curdex+=MAX_BAR_NUM
    return f_lines


def split_and_append_ima_arranged(abc_lines,abc_path,rmv_idx=None):
    f_lines = []
    header = []
    voice_lines = []
    in_header = True
    if(ONSET_DATA==None): #Only add labels of onset data is defined
        return abc_lines
    
    for line in abc_lines:
        if in_header and not line.startswith("[V:"):
            header.append(line)
        else:
            in_header = False
            if line.startswith("[V:"):
                voice_lines.append(line)
    #print(len(header),len(voice_lines))
    rpath = abc_path.replace(PATH_REPLACEMENT_TERM,"abc")
    labels = copy.copy(onset_data[rpath][TARGET_FEATURE])

    plabels = [[]]*len(labels)
    for key,value in labels.items():
        plabels[int(key)]=value
    labels=plabels
    if(rmv_idx and len(rmv_idx)>=1):
        for index in sorted(rmv_idx, reverse=True):
            del labels[index]
    assert len(labels)==len(voice_lines), f"Labels and Voice lines are of different lengths {len(labels),len(voice_lines)}"
    #print(labels)
    if(MAX_BAR_NUM==None):
        if(len(labels)!=len(voice_lines)):
            raise ValueError("Unequal label length",len(labels),len(voice_lines),abc_path)
        
        lines = [f"%{CONTROL_TERM} {' '.join([str(d) for d in label])}\n" for label in labels]

        return lines+abc_lines
    
    curdex = 0
    target = MIN_BAR_NUM
    while curdex<len(labels):
        if(MIN_BAR_NUM and (target-curdex)<MIN_BAR_NUM): #skip subsections of pieces that are too short
            break
        target = min(curdex+MAX_BAR_NUM,len(labels))
        sublabels = labels[curdex:target]
        lines = [f"%{CONTROL_TERM} {' '.join([str(d) for d in label])}\n" for label in labels]
        musicLines = lines+header+voice_lines[curdex:target]
        f_lines.append(musicLines)
        curdex+=MAX_BAR_NUM
    return f_lines



def split_and_append_labels(abc_lines,abc_path, rmv_idx=None):
    f_lines = []
    header = []
    voice_lines = []
    in_header = True
    if(ONSET_DATA==None): #Only add labels of onset data is defined
        return abc_lines
    
    for line in abc_lines:
        if in_header and not line.startswith("[V:"):
            header.append(line)
        else:
            in_header = False
            if line.startswith("[V:"):
                voice_lines.append(line)
    #print(len(header),len(voice_lines))
    rpath = abc_path.replace(PATH_REPLACEMENT_TERM,"abc")
    labels = copy.copy(onset_data[rpath][TARGET_FEATURE])
    if(type(labels)==dict):
        plabels = [[]]*len(labels)
        for key,value in labels.items():
            plabels[int(key)]=value
        
        labels=plabels
    #print(labels)
    if(rmv_idx and len(rmv_idx)>=1):
        for index in sorted(rmv_idx, reverse=True):
            del labels[index]

    #print("alignment",len(labels),len(voice_lines))
    assert len(labels)==len(voice_lines), f"Labels and Voice lines are of different lengths {len(labels),len(voice_lines)}"
    if(MAX_BAR_NUM==None):
        if(len(labels)!=len(voice_lines)):
            raise ValueError("Unequal label length",len(labels),len(voice_lines),abc_path)
        
        line = [f"%{CONTROL_TERM} {' '.join([str(d) for d in labels])}\n"]
        return [line+abc_lines]
    
    curdex = 0
    target = MIN_BAR_NUM
    while curdex<len(labels):
        if(MIN_BAR_NUM and (target-curdex)<MIN_BAR_NUM): #skip subsections of pieces that are too short
            break
        target = min(curdex+MAX_BAR_NUM,len(labels))
        sublabels = labels[curdex:target]
        line = [f"%{CONTROL_TERM} {' '.join([str(d) for d in sublabels])}\n"]
        musicLines = line+header+voice_lines[curdex:target]
        f_lines.append(musicLines)
        curdex+=MAX_BAR_NUM
    return f_lines
split_and_append_ima_arranged

def abc_preprocess_pipeline(abc_path):

    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()
    
    meter = None
    meter_matches = 0
    for entry in abc_lines:
        if(entry.startswith("M:")):
            rmatch = re.search(r"\d+/\d+",entry)
            if(rmatch):
                meter_matches+=1
                meter = rmatch.group()
    #print(meter)
    if meter_matches != 1:
        raise ValueError(f"{abc_path} contains {meter_matches} valid meters (expected exactly 1)")
    
    # delete blank lines
    abc_lines = [line for line in abc_lines if line.strip() != '']

    # unidecode
    abc_lines = unidecode_abc_lines(abc_lines)

    # clean information field
    abc_lines = remove_information_field(abc_lines=abc_lines, info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])

    # dele"""te bar number annotations
    abc_lines = remove_bar_no_annotations(abc_lines)
            
    # delete \"
    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        else:
            if r'\"' in line:
                abc_lines[i] = abc_lines[i].replace(r'\"', '')

    # delete text annotations with quotes
    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    # check bar alignment
    try:
        _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
        if not bar_no_equal_flag:
            raise ValueError((abc_path, 'Unequal bar number'))
    except:
        raise ValueError("check_alignment_unrotated failed")

    # deal with text annotations: remove too long text annotations; remove consecutive non-alphabet/number characters
    for i, line in enumerate(abc_lines):
        quote_matches = re.findall(r'"[^"]*"', line)
        for match in quote_matches:
            if match == '""':
                line = line.replace(match, '')
            if match[1] in ['^', '_']:
                sub_string = match
                pattern = r'([^a-zA-Z0-9])\1+'
                sub_string = re.sub(pattern, r'\1', sub_string)
                if len(sub_string) <= 40:
                    line = line.replace(match, sub_string)
                else:
                    line = line.replace(match, '')
        abc_lines[i] = line

    ori_abc_name = os.path.splitext(os.path.split(abc_path)[-1])[0]

    # transpose
    metadata_lines, part_text_dict = extract_metadata_and_parts(abc_lines)
    global_metadata_dict, local_metadata_dict = extract_global_and_local_metadata(metadata_lines)
    if global_metadata_dict['K'][0] == 'none':
        global_metadata_dict['K'][0] = 'C'
    ori_key = global_metadata_dict['K'][0]
    interleaved_abc = rotate_abc(abc_lines)
    interleaved_path = os.path.join(INTERLEAVED_FOLDER, ori_abc_name + '.abc')
    with open(interleaved_path, 'w') as w:
        w.writelines(interleaved_abc)

    abc_files = set()
    if(USE_AUGMENTED):
        keyiter = Key2index.keys()
    else:
        keyiter = [ori_key]
    #print(Key2index.keys(),[ori_key])
    for key in keyiter:
        if(USE_AUGMENTED):
            transposed_abc_text = transpose_an_abc_text(abc_lines, key)
            transposed_abc_lines = transposed_abc_text.split('\n')

        else:
            transposed_abc_lines = abc_lines
        
        transposed_abc_lines = list(filter(None, transposed_abc_lines))
        transposed_abc_lines = [line + '\n' for line in transposed_abc_lines]

        # rest reduction
        metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = \
            extract_barline_and_bartext_dict(transposed_abc_lines)
        reduced_abc_lines = metadata_lines

        removed_bars = {}
        voices = len(bar_text_dict)
        for i in range(len(bar_text_dict['V:1'])):
            line = ''
            for symbol in prefix_dict.keys():
                valid_flag = False
                
                for char in bar_text_dict[symbol][i]:
                    if char.isalpha() and not char in ['Z', 'z', 'X', 'x']:
                        valid_flag = True
                        break
                if valid_flag:
                    if i == 0:
                        part_patch = '[' + symbol + ']' + prefix_dict[symbol] + left_barline_dict[symbol][0] + bar_text_dict[symbol][0] + right_barline_dict[symbol][0]
                    else:
                        part_patch = '[' + symbol + ']' + bar_text_dict[symbol][i] + right_barline_dict[symbol][i]
                    line += part_patch
                if not valid_flag:
                    val = removed_bars.get(i)
                    if(val):
                        removed_bars.update({i:val+1})
                    else:
                        removed_bars.update({i:1})
                    
            line += '\n'
            reduced_abc_lines.append(line)
        removed_bars_ls = []
        #print(removed_bars, bar_text_dict.keys())
        for bar,removed in removed_bars.items():
            if(removed>=len(bar_text_dict)):#If all voices are removed adjust labels accordingly
                removed_bars_ls.append(bar)

        #print("removing from bar",removed_bars_ls)
        if(FEATURE_TYPE=="bar"):
            abc_split = split_and_append_labels(reduced_abc_lines,abc_path, removed_bars_ls)
        elif(FEATURE_TYPE=="onset"):
            abc_split = split_and_append_ima(reduced_abc_lines,abc_path,meter)
        elif(FEATURE_TYPE=="arranged"):
            abc_split = split_and_append_ima_arranged(reduced_abc_lines,abc_path, removed_bars_ls)
        for i in range(0,len(abc_split)):
            abc_name = f"{ori_abc_name}_{i}"
            abc_sub_lines = abc_split[i]
            #print(len(abc_lines),i)
            abc_files.add(abc_name)
            if(USE_AUGMENTED):
                reduced_abc_name = abc_name + '_' + key
                reduced_abc_path = os.path.join(AUGMENTED_FOLDER, key, reduced_abc_name + '.abc')
            else:
                reduced_abc_name = abc_name
                reduced_abc_path = os.path.join(AUGMENTED_FOLDER, reduced_abc_name + '.abc')
            #print(abc_sub_lines)
            abc_sub_lines = [s.replace("\n\n","\n") for s in abc_sub_lines]
            with open(reduced_abc_path, 'w', encoding='utf-8') as w:
                w.writelines(abc_sub_lines)

    return list(abc_files), ori_key





if __name__ == '__main__':
    data = [] 
    file_list = os.listdir(ORI_FOLDER)
    #file_list = file_list[:50]
    for file in tqdm(file_list):
        ori_abc_path = os.path.join(ORI_FOLDER, file)
        try:
            abc_names, ori_key = abc_preprocess_pipeline(ori_abc_path)
        except Exception as e:
            print(ori_abc_path, 'failed to pre-process.',e)
            continue
        for abc_name in abc_names:
            data.append({
                'path': os.path.join(AUGMENTED_FOLDER, abc_name),
                'key': ori_key
            })

    random.shuffle(data)
    eval_data = data[ : int(EVAL_SPLIT * len(data))]
    train_data = data[int(EVAL_SPLIT * len(data)) : ]

    data_index_path = AUGMENTED_FOLDER + '.jsonl'
    eval_index_path = AUGMENTED_FOLDER + '_eval.jsonl'
    train_index_path = AUGMENTED_FOLDER + '_train.jsonl'


    with open(data_index_path, 'w', encoding='utf-8') as w:
        for d in data:
            w.write(json.dumps(d) + '\n')
    with open(eval_index_path, 'w', encoding='utf-8') as w:
        for d in eval_data:
            w.write(json.dumps(d) + '\n')
    with open(train_index_path, 'w', encoding='utf-8') as w:
        for d in train_data:
            w.write(json.dumps(d) + '\n')

    