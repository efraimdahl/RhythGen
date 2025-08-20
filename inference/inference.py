import os
import time
import torch
from rhythgen.model import *
from rhythgen.model_config import *
from rhythgen.patchilizer import Patchilizer

from .if_config import *
from rhythgen.model_config import *
from transformers import GPT2Config, LlamaConfig
from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re, Barline_regexPattern
from abctoolkit.transpose import Note_list, Pitch_sign_list
from abctoolkit.duration import calculate_bartext_duration
from safetensors.torch import load_file
Note_list = Note_list + ['z', 'x']

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.makedirs(ORIGINAL_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INTERLEAVED_OUTPUT_FOLDER, exist_ok=True)

patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                          max_length=PATCH_LENGTH,
                          max_position_embeddings=PATCH_LENGTH,
                          n_embd=HIDDEN_SIZE,
                          num_attention_heads=HIDDEN_SIZE // 64,
                          vocab_size=1)
byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS,
                         max_length=PATCH_SIZE + 1,
                         max_position_embeddings=PATCH_SIZE + 1,
                         hidden_size=HIDDEN_SIZE,
                         num_attention_heads=HIDDEN_SIZE // 64,
                         vocab_size=128)


con_config = ControlConfig(control_dim=1,
                           embed_dim=HIDDEN_SIZE,
                           nhead=ENCODER_HEADS,
                           nlayer=ENCODER_LAYERS,
                           grid_size=GRID_SIZE,
                           intermediate_dim=256,
                           )
model = RhythGenModel(encoder_config=patch_config, decoder_config=byte_config, control_config=con_config)

print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# Manually assign random weights to lm_head before loading
if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
    model.lm_head.weight = model.transformer.wte.weight.clone().detach()  # Assign random but same shape

# Load safetensors weights
checkpoint = load_file(INFERENCE_WEIGHTS_PATH)

# Load state dict with strict=False to avoid missing key errors
missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

print("Missing keys",missing_keys)
print("Unexpected Keyws",unexpected_keys)

# Manually re-tie weights after loading
if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
    model.lm_head.weight = model.transformer.wte.weight  # Ensure correct shared weight

# Move to device and set to eval mode
model = model.to(device)
model.eval()




def rest_unreduce(abc_lines):

    tunebody_index = None
    for i in range(len(abc_lines)):
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[: tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    for line in metadata_lines:
        if line.startswith('%%score'):
            for round_bracket_match in re.findall(r'\((.*?)\)', line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith('V:'):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])
    z_symbol_list = []  # voices that use z as rest
    x_symbol_list = []  # voices that use x as rest
    for voice_group in voice_group_list:
        z_symbol_list.append('V:' + voice_group[0])
        for j in range(1, len(voice_group)):
            x_symbol_list.append('V:' + voice_group[j])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''

        line = re.sub(r'^\[r:[^\]]*\]', '', line)

        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)

        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            line_bar_dict[key] = value

        # calculate duration and collect barline
        dur_dict = {}  
        for symbol, bartext in line_bar_dict.items():
            right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext = bartext[:-len(right_barline)]
            try:
                bar_dur = calculate_bartext_duration(bartext)
            except:
                bar_dur = None
            if bar_dur is not None:
                if bar_dur not in dur_dict.keys():
                    dur_dict[bar_dur] = 1
                else:
                    dur_dict[bar_dur] += 1

        try:
            ref_dur = max(dur_dict, key=dur_dict.get)
        except:
            pass    # use last ref_dur

        if i == 0:
            prefix_left_barline = line.split('[V:')[0]
        else:
            prefix_left_barline = ''

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
            else:
                if symbol in z_symbol_list:
                    symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + right_barline
                elif symbol in x_symbol_list:
                    symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + right_barline
            unreduced_line += '[' + symbol + ']' + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + '\n')

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines



def inference_patch(og_prompt_lines=[], pieces=NUM_SAMPLES, prefix="",guidance_scale = CFG_GUIDANCE, piece_count=0):

    file_no = 1
    failures = 0
    bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

    if(COND_FORMAT=="cat"):
        null_control = PADDING_INDEX
    if(COND_FORMAT=="con"):
        null_control = [0]*GRID_SIZE
    
    while file_no <= pieces:
        if(failures>5):
            print("Too many consecutive failures")
            return
        start_time = time.time()
        start_time_format = time.strftime("%Y%m%d-%H%M%S")
        
        prompt_start_index = 0 #separate prompt from control
        controls = []
        print(f"Generating: {file_no+piece_count}/{NUM_SAMPLES} with guidance scale {guidance_scale}")
        
        if(len(og_prompt_lines)>0):
            while(og_prompt_lines[prompt_start_index].startswith("%"+CONTROL_TERM)):
                if(COND_FORMAT=="cat"):
                    controls = og_prompt_lines[prompt_start_index].split(" ")[1:]
                    controls= [int(control.strip()) for control in controls]
                    break
                elif(COND_FORMAT=="con"):
                    sub_controls_s = og_prompt_lines[prompt_start_index].split(" ")[1:]
                    sub_controls = []
                    for control_val in sub_controls_s:
                        try:
                            sub_controls.append(float(control_val))
                        except ValueError:
                            sub_controls.append(0)
                    sub_controls = sub_controls + [0]*(GRID_SIZE-len(sub_controls)) #Pad controls
                    controls.append(sub_controls)
                prompt_start_index+=1
            #print("Prompt Patches: ", prompt_start_index)
            prompt_lines = og_prompt_lines[prompt_start_index:]
            controls = [null_control]+controls

        else:
            prompt_lines = []
        
        prompt_patches, _ = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        #print(''.join(byte_list), end='')
        #print(prompt_patches)
        
        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)
        #Create 0 controls for metadata
        control_patches = [null_control]*len(prompt_patches)
        if(COND_FORMAT=="cat"):
            control_patches  = torch.tensor(control_patches, dtype=torch.long, device=device)
        elif(COND_FORMAT=="con"):
            control_patches  = torch.tensor(control_patches, dtype=torch.float, device=device)
        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

        if(COND_FORMAT=="cat"):
            controls = torch.tensor(controls, dtype=torch.long)
        elif(COND_FORMAT=="con"):
            
            controls = torch.tensor(controls, dtype=torch.float)
        #print("Input Patches: ", input_patches.shape, control_patches)
        failure_flag = False
        end_flag = False
        cut_index = None

        tunebody_flag = False

        current_control_idx = 0
        current_control = controls[current_control_idx]

        #print(controls)
        vpatch = False
        v_control_flag = True
        while True:
           #ncond_controls = torch.full_like(control_patches, fill_value=PADDING_INDEX)

            predicted_patch, vpatch = model.generate_cfg(input_patches.unsqueeze(0),
                                             control_patches.unsqueeze(0),
                                             top_k=TOP_K,
                                             top_p=TOP_P,
                                             temperature=TEMPERATURE,
                                             guidance_scale=guidance_scale,
                                             vpatch=vpatch, 
                                             current_control=current_control)
            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                tunebody_flag = True
                #print("Forcing r0")
                r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                r0_control = torch.tensor(controls[1]).unsqueeze(0).to(device) 
                current_control_idx+=1
                current_control = controls[current_control_idx]
                tmp_control_patches = torch.concat([control_patches,r0_control])
                predicted_patch, vpatch = model.generate_cfg(temp_input_patches.unsqueeze(0),
                                                 control_patches.unsqueeze(0),
                                                 top_k=TOP_K,
                                                 top_p=TOP_P,
                                                 temperature=TEMPERATURE, 
                                                 guidance_scale=guidance_scale,
                                                 vpatch=vpatch, current_control=current_control)
                predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
            
            if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                end_flag = True
                break
            
            next_patch = patchilizer.decode([predicted_patch])
            
            """            if(tunebody_flag and next_patch.endswith('\n')): #reactivate controls for first voice
                if(V_CONTROL):
                    v_control_flag=True
                #print("Newline")

                current_control_idx += 1
            
            elif(V_CONTROL and tunebody_flag and "[V:1]" not in next_patch and "[V:" in next_patch): #deactivate controls for first voice
                v_control_flag = False
            """

            for char in next_patch:
                byte_list.append(char)
                #print(char, end='')
            #print(vpatch,control_patches)
            
            #print(" ", current_control_idx, control_patches[-1],v_control_flag)

            patch_end_flag = False
            
            for j in range(len(predicted_patch)):
                if patch_end_flag:
                    predicted_patch[j] = patchilizer.special_token_id
                if predicted_patch[j] == patchilizer.eos_token_id:
                    patch_end_flag = True

            if current_control_idx < len(controls):
                current_control = controls[current_control_idx]
            else:
                current_control = controls[-1]  # Or PAD if exceeding


            if(COND_FORMAT=="con"):
                current_control_enc  = torch.tensor(current_control, dtype=torch.float).unsqueeze(0)
                #print("Control Shape", current_control_enc.shape, control_patches.shape)
            else:
                current_control_enc = torch.tensor([current_control], device=device)
            predicted_patch = torch.tensor([predicted_patch], device=device)  # (1, 16)

            current_control_enc = current_control_enc.to(device)
            control_patches = control_patches.to(device)
            #print(current_control)
            #print(control_patches.shape, current_control_enc.shape,input_patches.shape,predicted_patch.shape)
            control_patches = torch.cat([control_patches, current_control_enc], dim=0)
            input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

            if len(byte_list) > 102400:  
                print("Too long generation (bytes) ")
                failure_flag = True
                failures+=1
                break
            if time.time() - start_time > 3 * 60:  
                print("Too long generation (time)")
                failure_flag = True
                failures+=1
                break

            if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
                failure_flag = True
                
                abc_code = ''.join(byte_list)
                abc_lines = abc_code.split('\n')

                tunebody_index = None
                for i, line in enumerate(abc_lines):
                    if line.startswith('[r:') or line.startswith('[V:'):
                        tunebody_index = i
                        break
                if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                    break

                metadata_lines = abc_lines[:tunebody_index]
                tunebody_lines = abc_lines[tunebody_index:]

                metadata_lines = [line + '\n' for line in metadata_lines]
                if not abc_code.endswith('\n'):  
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [
                        tunebody_lines[-1]]
                else:
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]

                if cut_index is None:
                    cut_index = len(tunebody_lines) // 2

                abc_code_slice = ''.join(metadata_lines + tunebody_lines[-cut_index:])
                input_patches = patchilizer.encode_generate(abc_code_slice)

                input_patches = [item for sublist in input_patches for item in sublist]
                input_patches = torch.tensor([input_patches], device=device)
                input_patches = input_patches.reshape(1, -1)


        if not failure_flag:
            generation_time_cost = time.time() - start_time

            abc_text = ''.join(byte_list)
            filename = f"{prefix}-{format(generation_time_cost)}_{str(file_no)}.abc"
            # unreduce
            print(f"Saving {filename}")
            abc_lines = abc_text.split('\n')
            abc_lines = list(filter(None, abc_lines))
            abc_lines = [line + '\n' for line in abc_lines]
            #print(abc_lines)


            unreduced_output_path = os.path.join(INTERLEAVED_OUTPUT_FOLDER, filename)
            
            try:
                abc_lines = rest_unreduce(abc_lines)

                with open(unreduced_output_path, 'w') as file:
                    file.writelines(abc_lines)
                failures=0
            except Exception as e:
                print("Failed to Save",e)
                failures+=1
                pass
            else:
                # original
                original_output_path = os.path.join(ORIGINAL_OUTPUT_FOLDER, filename)
                with open(original_output_path, 'w') as w:
                    w.write(abc_text)

                file_no += 1
        
        else:
            print('failed')
        


inference_prefix = ["%%score 1 \n","L:1/8\n","Q:1/4=95\n","M:4/4\n","K:A\n",
                    'V:1 treble nm="Piano"\n','[V:1]\n']

if __name__ == '__main__':
    guidance_scales = CFG_GUIDANCE
    piece_count = STARTING_CONDITION[1]
    
    if(PROMPT_PATH.endswith("txt")):
        with open(PROMPT_PATH,"r") as infile:
            lines = infile.readlines()
        split = int(NUM_SAMPLES/len(lines))
        piece_count=0
        for meter in ["4/4","2/4","2/2"]:
            inference_prefix[3] = f"M:{meter}\n"
            print(f"Generating in {meter} with",inference_prefix)
            for guidance_scale in guidance_scales[STARTING_CONDITION[0]:]:
                for i in range(0,len(lines)):
                    cur_line = lines[i].strip()
                    if(cur_line==""):
                        print("Reached Empty prompt line")
                        break
                    prompt_lines = [cur_line]+inference_prefix
                    try:
                        inference_patch(og_prompt_lines=prompt_lines,pieces=split,
                                        prefix=f"sync-{guidance_scale}-{meter[0]}{meter[2]}-{i}"
                                        ,guidance_scale=guidance_scale,piece_count=piece_count)
                        piece_count+=split
                    except Exception as e:
                        print("Failed to Generate",e)
                        
    elif(PROMPT_PATH):
        abs_path = os.path.abspath(PROMPT_PATH)
        file_list = []
        for path, dirs, files in os.walk(abs_path):
            for file in files:
                if file.lower().endswith(".abc"):
                    file_list.append(file)
        print(abs_path)
        print(len(file_list),NUM_SAMPLES)
        split = int(NUM_SAMPLES/len(file_list))
        
        file_list.sort()
        
        for guidance_scale in guidance_scales[STARTING_CONDITION[0]:]:
            starting_piece = int(piece_count/split)
            for file in file_list[STARTING_CONDITION[1]:]:
                prompt_lines = []

                try:
                    with open(os.path.join(PROMPT_PATH,file),"r") as infile:
                        current_line = infile.readline()
                        while(not current_line.startswith(f"[V:")):
                            prompt_lines.append(current_line)
                            current_line = infile.readline()
                        #print(prompt_lines)
                        #prompt_lines.append("%%score 1 { ( 2 4 ) | ( 3 5 ) }\n")
                    inference_patch(og_prompt_lines=prompt_lines,
                                    pieces=split,
                                    prefix=f"sync-{guidance_scale}-{file}", 
                                    guidance_scale=guidance_scale,
                                    piece_count=piece_count)
                    piece_count+=split
                except Exception as e:
                    print("Failed to Generate",e)
                pass
            piece_count=0
    """
    
    else:
        split = NUM_SAMPLES/6
        inference_patch(og_prompt_lines=[f"%{CONTROL_TERM} 1\n","%%score 1 { ( 2 4 ) | ( 3 5 ) }\n"
], pieces=split, prefix="density_1")
        inference_patch(og_prompt_lines=[f"%{CONTROL_TERM} 2\n", "%%score 1 { ( 2 4 ) | ( 3 5 ) }\n"
], pieces=split, prefix="density_2")
        inference_patch(og_prompt_lines=[f"%{CONTROL_TERM} 3\n", "%%score 1 { ( 2 4 ) | ( 3 5 ) }\n"
], pieces=split, prefix="density_3")
        inference_patch(og_prompt_lines=[f"%{CONTROL_TERM} 4\n","%%score 1 { ( 2 4 ) | ( 3 5 ) }\n"
], pieces=split, prefix="density_4")
        inference_patch(og_prompt_lines=[f"%{CONTROL_TERM} 5\n","%%score 1 { ( 2 4 ) | ( 3 5 ) }\n"
], pieces=split, prefix="density_5")
        inference_patch(og_prompt_lines=[f"%{CONTROL_TERM} 6\n","%%score 1 { ( 2 4 ) | ( 3 5 ) }\n"
], pieces=split, prefix="density_6")
    """