import torch
import random
import bisect
import json
import re
from preprocessing.data_config import *
from .model_config import *

class Patchilizer:
    def __init__(self, stream=PATCH_STREAM):
        self.stream = stream
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0
        
        if(COND_FORMAT=="cat"):
            self.null_control = PADDING_INDEX
        if(COND_FORMAT=="con"):
            self.null_control = [0]*GRID_SIZE

    def split_bars(self, body_lines, controls):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        extended_controls = []
        #print("ControlLength:",len(body_lines),len(controls))
        try:
            for i in range(0,len(body_lines)):
                line = body_lines[i]
                #print(line, controls)

                current_control = controls[i]
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []
                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]  
                        new_line_bars = new_line_bars[:-1]
                
                #Mask out unwanted controls
                if(V_CONTROL):
                    new_line_controls= [self.null_control]*len(new_line_bars)
                    add_flag = False
                    for idx,voice in enumerate(new_line_bars):
                        #print(voice)
                        if(V_CONTROL in voice):
                            new_line_controls[idx]=current_control
                            #print("Added")
                else:
                    new_line_controls = [current_control]*len(new_line_bars)
                extended_controls+=new_line_controls
                new_bars += new_line_bars
        except Exception as e:
            print("splitting bar problems")
            print(e)
            pass

        return new_bars, extended_controls

    def split_patches(self, abc_text, patch_size=PATCH_SIZE, generate_last=False):
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i : i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def patch2chars(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes
        

    def patchilize_metadata(self, metadata_lines):

        metadata_patches = []
        for line in metadata_lines:
            metadata_patches += self.split_patches(line)
        metadata_controls=[self.null_control]*len(metadata_patches)
        return metadata_patches, metadata_controls
    
    def patchilize_tunebody(self, tunebody_lines, controls, encode_mode='train'):
        #print("Controls",controls)
        tunebody_patches = []
        bars, controls_per_bar = self.split_bars(tunebody_lines, controls)
        controls_per_patch = []
        if encode_mode == 'train':
            for i in range(0,len(bars)):
                current_control = controls_per_bar[i]
                new_patches = self.split_patches(bars[i])
                #print("Bar",bars[i],"-", new_patches)
                tunebody_patches += new_patches
                controls_per_patch += [current_control]*len(new_patches)

        elif encode_mode == 'generate':
            for i in range(0, len(bars)-1):
                current_control = controls_per_bar[i]
                new_patches = self.split_patches(bars[i])
                tunebody_patches += new_patches
                controls_per_patch += [current_control]*len(new_patches)
            tunebody_patches += self.split_patches(bars[-1], generate_last=True)
            if(COND_FORMAT=="con"):
                last_control = controls_per_bar[-1]
            else:
                last_control = controls_per_bar[-1]
            controls_per_patch.append(last_control)
        #assert len(tunebody_patches)==len(controls_per_patch)
        return tunebody_patches, controls_per_patch
    
    def extend_controls(self,controls,n_lines):
        """
        Extends the controls to match n_lines
        """
        if(len(controls)==0):
            return []
        if(n_lines>len(controls)):
            if(COND_FORMAT=="cat"):
                extended_controls = [controls[-1]]*(n_lines-len(controls))
                controls = controls+extended_controls
            else:
                extended_controls = [controls[-1]]*(n_lines-len(controls))
                controls = controls+extended_controls
        
        return controls
    
    def encode_train(self, abc_text, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True, cut=True):
        
        control_index = 0
        controls = []

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]
        

        while(lines[control_index].startswith("%"+CONTROL_TERM)):
            if(COND_FORMAT=="cat"):
                controls = lines[control_index].split(" ")[1:]
                controls= [int(control.strip()) for control in controls]
                break
            elif(COND_FORMAT=="con"):
                sub_controls_s = lines[control_index].split(" ")[1:]
                sub_controls = []
                for control_val in sub_controls_s:
                    try:
                        sub_controls.append(float(control_val))
                    except ValueError:
                        sub_controls.append(0)
                if(len(sub_controls)>GRID_SIZE):
                    sub_controls=sub_controls[0:GRID_SIZE]
                else:
                    sub_controls = sub_controls + [0]*(GRID_SIZE-len(sub_controls)) #Pad controls
                controls.append(sub_controls)
            control_index+=1
                
        #print(controls)
        tunebody_index = -1
        
        
        for i, line in enumerate(lines):
            if '[V:' in line:
                tunebody_index = i
                break
        metadata_lines = lines[control_index : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]
        controls = self.extend_controls(controls,len(tunebody_lines))

        if self.stream:
            tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                                enumerate(tunebody_lines)]    
        
        
        metadata_patches, metadata_controls  = self.patchilize_metadata(metadata_lines)
        tunebody_patches, tunebody_controls = self.patchilize_tunebody(tunebody_lines, controls, encode_mode='train')

        #Replace controls with padding during control dropout. 
        
        #print("Tunebody", tunebody_patches,tunebody_controls)
        #print("Metadata", metadata_patches,metadata_controls)

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)
            

            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]
            metadata_controls = [self.null_control] + metadata_controls
            tunebody_controls = tunebody_controls + [self.null_control]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if '\n' in patch]
                line_index_for_cut_index = list(range(len(available_cut_indexes)))  
                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index) 
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                    patch_controls = metadata_controls + tunebody_controls[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index] 
                    stream_tunebody_lines = tunebody_lines[line_index : ]
                    stream_control_per_line = controls[line_index : ]
                    
                    stream_tunebody_patches, stream_tunebody_controls = self.patchilize_tunebody(stream_tunebody_lines,stream_control_per_line, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                        stream_tunebody_controls = stream_tunebody_controls + [self.null_control]
                    patches = metadata_patches + stream_tunebody_patches
                    patch_controls = metadata_controls + stream_tunebody_controls
            else:
                patches = metadata_patches + tunebody_patches
                patch_controls = metadata_controls+tunebody_controls
        else:
            patches = metadata_patches + tunebody_patches
            patch_controls = metadata_controls+tunebody_controls

        if cut: 
            patches = patches[ : patch_length]
            patch_controls = patch_controls[:patch_length]
        else:  
            pass
        #for patch in patches:
            #print("Patch",patch)
        # encode to ids
        id_patches = []
        assert len(patches)==len(patch_controls),f"Patches and Controls are not matched {len(patches)} vs {len(patch_controls)} "

        #print(patch_controls)
        for i,patch in enumerate(patches):
            #print(patch,patch_controls[i])
            id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches, patch_controls

    def encode_generate(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True):

        lines = abc_code.split('\n')
        lines = list(filter(None, lines))
    
        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:') or line.startswith('[r:'):
                tunebody_index = i
                break
    
        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]   
    
        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'): 
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]
        
        if(COND_FORMAT=="cat"):
            null_control_patch = PADDING_INDEX
        elif(COND_FORMAT=="con"):
            null_control_patch = [0]*GRID_SIZE
        
        metadata_patches, controls = self.patchilize_metadata(metadata_lines)
        tunebody_patches, tunebody_controls = self.patchilize_tunebody(tunebody_lines, controls, encode_mode='generate')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)

            metadata_patches = [bos_patch] + metadata_patches
    
        patches = metadata_patches + tunebody_patches
        patches = patches[ : patch_length]

        # encode to ids
        id_patches = []
        patch_control = []
        for patch in patches:
            patch_control.append(null_control_patch)
            if len(patch) < PATCH_SIZE and patch[-1] != chr(self.eos_token_id):
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)
        
        return id_patches, patch_control

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2chars(patch) for patch in patches)

