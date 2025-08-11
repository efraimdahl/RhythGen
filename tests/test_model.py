
import os
import time
import torch
from rhythgen.patchilizer import Patchilizer
from rhythgen.model import ControlConfig, RhythGenModel
from rhythgen.utils import load_composite_state_dict
from rhythgen.model_config import *
from data.data_config import *
from transformers import GPT2Config

PRETRAINED_PATH = '../Pretrained/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth'

CONTROL_ENCODER_PATH = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

for name, param in model.named_parameters():
    if "gate_scale" in name:
        print(name, param.shape)


print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

load_composite_state_dict(model,PRETRAINED_PATH, from_torch=True)

if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
    print("retied weight")
    model.lm_head.weight = model.transformer.wte.weight  # Ensure correct shared weight

print(model)
# Move to device and set to eval mode
model = model.to(device)
model.eval()


def inject_control(patch,control):
    subpatches = re.split("]",patch,maxsplit=1)
    if(len(subpatches)>0): 
        components = re.split(':|/', subpatches[0])
        if(len(components)>=3):
            rpatch= f"{components[0]}:{components[1]}/{components[2]}:{control}]"
    else:
        print("Malformed bar-break patch:",patch)



def test_patchilizer(example_path):
        with open(example_path, 'r', encoding='utf-8') as f:
            abc_text = f.read()
        
        file_bytes, controls = patchilizer.encode_train(abc_text)
        #print(controls)
        #print(file_bytes,contNotaGen/finetune/lc4919673_0_A.abcrols)
        file_masks = [1] * len(file_bytes)

        file_bytes = torch.tensor(file_bytes, dtype=torch.long).unsqueeze(0).to(device)
        file_masks = torch.tensor(file_masks, dtype=torch.long).unsqueeze(0).to(device)
        #for i, control in enumerate(controls):
            #print(i,control)
        controls = torch.tensor(controls, dtype=torch.float)
        controls.unsqueeze(0).to(device)
        
        print("Shape Controls:", controls.shape, "  Shape Input:",file_bytes.shape)



def run_inference_on_example(example_path):
    model.eval()
    with torch.no_grad():
        with open(example_path, 'r', encoding='utf-8') as f:
            abc_text = f.read()

        file_bytes, controls = patchilizer.encode_train(abc_text)
        file_masks = [1] * len(file_bytes)

        file_bytes = torch.tensor(file_bytes, dtype=torch.long).unsqueeze(0).to(device)
        file_masks = torch.tensor(file_masks, dtype=torch.long).unsqueeze(0).to(device)

        if(COND_FORMAT=="cat"):
            controls = torch.tensor(controls, dtype=torch.long).unsqueeze(0).to(device)
        elif(COND_FORMAT=="con"):
            controls = torch.tensor(controls, dtype=torch.float).unsqueeze(0).to(device)
        print("Control Sum:",sum(sum(controls)))
        print("Mask Sum:",sum(sum(file_masks)))
        print("Control Shape", controls.shape)
            
        print(f"Patches {file_bytes.shape}, Masks {file_masks.shape}, Control:{controls.shape}")
        print(f"Running inference on: {example_path}")
        output = model(file_bytes, file_masks, controls, epoch=1)
        logits = output.logits  # Shape: [batch_size, seq_len, vocab_size]
        print("Logits shape:", logits.shape)
        print("Model Loss", output.loss)
        #print("Sample logits:", logits[0, :5])  # print logits for first 5 tokens



if __name__ == '__main__':
    example_file = "test_song_vec.abc"
    run_inference_on_example(example_file)
