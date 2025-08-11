


def inference_patch(og_prompt_lines=[], pieces=NUM_SAMPLES, prefix=""):

    file_no = 1

    bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

    while file_no <= pieces:

        start_time = time.time()
        start_time_format = time.strftime("%Y%m%d-%H%M%S")

        prompt_start_index = 0 #separate prompt from control
        controls = ["0"]

        if(len(og_prompt_lines)>0):
            if(og_prompt_lines[0].startswith("%"+CONTROL_TERM)):
                prompt_start_index=1
                controls += og_prompt_lines[0].split(" ")[1:]
                controls = [int(control.strip()) for control in controls] 
            print(controls, file_no,pieces,NUM_SAMPLES)
            prompt_lines=og_prompt_lines[prompt_start_index:]
        else:
            prompt_lines = []
    
        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        print(''.join(byte_list), end='')

        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)
        control_patches = [PADDING_INDEX]*len(prompt_patches)

        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)
        
        control_patches = torch.tensor(control_patches, dtype=torch.long)

        print("Input Patches: ", input_patches.shape, control_patches)
        failure_flag = False
        end_flag = False
        cut_index = None

        tunebody_flag = False

        current_control_idx = 0
        
        while True:
            ncond_controls = torch.full_like(control_patches, fill_value=PADDING_INDEX)

            predicted_patch = model.generate(input_patches.unsqueeze(0),
                                             ncond_controls.unsqueeze(0),
                                             top_k=TOP_K,
                                             top_p=TOP_P,
                                             temperature=TEMPERATURE)
            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                tunebody_flag = True
                print("Forcing r0")
                r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                r0_control = torch.tensor(controls[1]).unsqueeze(0).to(device) 
                current_control_idx+=1

                
                tmp_control_patches = torch.concat([control_patches,r0_control])
                predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                 ncond_controls.unsqueeze(0),
                                                 top_k=TOP_K,
                                                 top_p=TOP_P,
                                                 temperature=TEMPERATURE)
                predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
            if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                end_flag = True
                break
            
            next_patch = patchilizer.decode([predicted_patch])
            
            if(tunebody_flag and next_patch.endswith('\n')):
                #print("Newline")
                current_control_idx += 1
            
            for char in next_patch:
                byte_list.append(char)
                print(char, end='')
            
            #print(" ", control_patches[-1])

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
            current_control_enc = torch.tensor([current_control], device=device)
            predicted_patch = torch.tensor([predicted_patch], device=device)  # (1, 16)

            #print(control_patches.shape, current_control_enc.shape,input_patches.shape,predicted_patch.shape)
            control_patches = torch.cat([control_patches, current_control_enc], dim=0)
            input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)
            

            if len(byte_list) > 102400:  
                failure_flag = True
                break
            if time.time() - start_time > 20 * 60:  
                failure_flag = True
                break

            if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
                print('Stream generating...')
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
            abc_lines = abc_text.split('\n')
            abc_lines = list(filter(None, abc_lines))
            abc_lines = [line + '\n' for line in abc_lines]

