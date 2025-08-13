import os
import gc
import time
import math
import json
import wandb
import torch
import random
import numpy as np
from abctoolkit.transpose import Key2index, Key2Mode
from rhythgen.patchilizer import Patchilizer
from rhythgen.model import *
from rhythgen.model_config import *
from rhythgen.utils import load_composite_state_dict
from preprocessing.data_config import *
from .ft_config import *
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, LlamaConfig, get_scheduler, get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import load_file

Index2Key = {index: key for key, index in Key2index.items() if index not in [1, 11]}
Mode2Key = {mode: key for key, mode_list in Key2Mode.items() for mode in mode_list }

# Set up distributed training
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1 or torch.cuda.device_count() > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl')# if world_size > 1 else None

else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
# Set random seed
seed = 0 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = BATCH_SIZE

patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                    max_length=PATCH_LENGTH, 
                    max_position_embeddings=PATCH_LENGTH,
                    n_embd=HIDDEN_SIZE,
                    num_attention_heads=HIDDEN_SIZE//64,
                    vocab_size=1)
char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, 
                            max_length=PATCH_SIZE+1, 
                            max_position_embeddings=PATCH_SIZE+1,
                            hidden_size=HIDDEN_SIZE,
                            num_attention_heads=HIDDEN_SIZE//64,
                            vocab_size=128)

    #def __init__(self, control_dim=1, embed_dim=768, nhead=4, intermediate_dim=3072, grid_size=48, **kwargs):

con_config = ControlConfig(control_dim=1,
                           embed_dim=HIDDEN_SIZE,
                           nhead=ENCODER_HEADS,
                           nlayer=ENCODER_LAYERS,
                           grid_size=GRID_SIZE,
                           intermediate_dim=256,
                           )



model = RhythGenModel(encoder_config=patch_config, decoder_config=char_config, control_config=con_config)

model = model.to(device)

if FREEZE:
    ignore_layers = [
        "control_embedding", "control_projection",
        "control_q_proj", "control_k_proj", "control_v_proj",
        "q_gate_scale", "k_gate_scale", "v_gate_scale","grid_attn_proj",
    ]
    
    for name, param in model.named_parameters():
        # Check if any of the ignore_layers are substrings of the parameter name
        if not any(layer in name for layer in ignore_layers):
            param.requires_grad = False

# Calculate parameter stats
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_percent = 100.0 * trainable_params / total_params

print(f"Trainable Parameters: {trainable_params} / {total_params} ({trainable_percent:.2f}%)")

if world_size > 1 or torch.cuda.device_count() > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def clear_unused_tensors():
    gc.disable()  # Temporarily disable garbage collection
    try:
        # Get the set of tensor ids used by the model
        if hasattr(model, "module"):
            model_tensors = {id(p) for p in model.module.parameters()}
        else:
            model_tensors = {id(p) for p in model.parameters()}
        
        # Get the set of tensor ids used by the optimizer
        optimizer_tensors = {
            id(state) 
            for state_dict in optimizer.state.values() 
            for state in state_dict.values()
            if isinstance(state, torch.Tensor)  # Ensure only tensors are considered
        }

        # List of all CUDA tensors currently in memory
        tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]
        
        # Create weak references to avoid interfering with garbage collection
        tensor_refs = [weakref.ref(tensor) for tensor in tensors]

        for tensor_ref in tensor_refs:
            tensor = tensor_ref()  # Dereference the weak reference
            if tensor is not None and id(tensor) not in model_tensors and id(tensor) not in optimizer_tensors:
                # Mark the tensor for deletion
                tensor.detach_()  # Detach from computation graph
                del tensor  # Delete the tensor reference
    except:
        pass

    finally:
        gc.enable()  # Re-enable garbage collection
        gc.collect()  # Force a garbage collection
        torch.cuda.empty_cache()  # Clear the CUDA cache

def collate_batch(input_batches):
    
    input_patches, input_masks, input_conditions = zip(*input_batches)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)
    input_conditions = torch.nn.utils.rnn.pad_sequence(input_conditions, batch_first=True, padding_value=0)

    return input_patches.to(device), input_masks.to(device), input_conditions.to(device)

def split_into_minibatches(input_patches, input_masks, input_controls, minibatch_size):
    minibatches = []
    for start_idx in range(0, len(input_patches), minibatch_size):
        end_idx = start_idx + minibatch_size
        minibatch_patches = input_patches[start_idx:end_idx]
        minibatch_masks = input_masks[start_idx:end_idx]
        minibatch_controls = input_controls[start_idx:end_idx]
        minibatches.append((minibatch_patches, minibatch_masks, minibatch_controls))
    return minibatches

class NotaGenDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx, transpose=USE_AUGMENTED):
        
        filepath = self.filenames[idx]['path']
        ori_key = Mode2Key[self.filenames[idx]['key']]
        if(transpose):
            # choose a key to transpose, according to a probility distribution
            ori_key_index = Key2index[ori_key]
            available_index = [(ori_key_index + offset) % 12 for offset in range(-3, 4)]
            index_prob = [1/16, 2/16, 3/16, 4/16, 3/16, 2/16, 1/16]
            index_prob_range = [0] + [sum(index_prob[0 : i + 1]) for i in range(len(index_prob))]
            random_number = random.random()
            for i in range(len(index_prob_range) - 1):
                if index_prob_range[i] <= random_number < index_prob_range[i + 1]:
                    des_key_index = available_index[i]
            if des_key_index == 1:
                des_key = 'Db' if random.random() < 0.8 else 'C#'   
            elif des_key_index == 11:
                des_key = 'B' if random.random() < 0.8 else 'Cb'
            elif des_key_index == 6:
                des_key = 'F#' if random.random() < 0.5 else 'Gb'
            else:
                des_key = Index2Key[des_key_index]
            folder = os.path.dirname(filepath)
            name = os.path.split(filepath)[-1]
            des_filepath = os.path.join(folder, des_key, name + '_' + des_key + '.abc')
        else:
            des_filepath = os.path.join(filepath + '.abc')
        with open(des_filepath, 'r', encoding='utf-8') as f:
            abc_text = f.read()

        file_bytes, controls = patchilizer.encode_train(abc_text)
        file_masks = [1] * len(file_bytes)

        file_bytes = torch.tensor(file_bytes, dtype=torch.long)
        file_masks = torch.tensor(file_masks, dtype=torch.long)
        
        #Apply Control Dropout
        
        if(COND_FORMAT=="cat"):
            controls = torch.tensor(controls, dtype=torch.long)
        elif(COND_FORMAT=="con"):
            controls = torch.tensor(controls, dtype=torch.float)
        
        if(random.random()<P_CONTROL_DROPOUT):
            controls = torch.full_like(controls, fill_value=PADDING_INDEX)
            
        return file_bytes, file_masks, controls


def process_one_batch(batch, epoch):
    input_patches, input_masks, controls = batch
    loss = model(input_patches, input_masks, controls, epoch=epoch).loss

    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss


# do one epoch for training
def train_epoch(epoch):
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()
    train_steps = (epoch-1)*len(train_set)

    for binput, bmask, bcontrol in tqdm_train_set:
        minibatches = split_into_minibatches(binput, bmask, bcontrol, BATCH_SIZE//ACCUMULATION_STEPS)
        for minibatch in minibatches:
            with autocast():
                loss = process_one_batch(minibatch, epoch) / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            total_train_loss += loss.item()
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        tqdm_train_set.set_postfix({str(global_rank)+'_train_loss': total_train_loss / iter_idx})
        train_steps += 1

        # Log the training loss to wandb
        if global_rank==0 and WANDB_LOGGING:
            wandb.log({"train_loss": total_train_loss / iter_idx}, step=train_steps)

        iter_idx += 1
        if iter_idx % 100 == 0:
            clear_unused_tensors()
        
    return total_train_loss / (iter_idx-1)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    total_eval_bpb = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        minibatches = split_into_minibatches(batch[0], batch[1], batch[2], BATCH_SIZE//ACCUMULATION_STEPS)
        for minibatch in minibatches:
            with torch.no_grad():
                loss = process_one_batch(minibatch, epoch) / ACCUMULATION_STEPS
            total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1
    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":
    
    # Initialize wandb
    if WANDB_LOGGING and global_rank==0:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="notagen",
                   name=WANDB_NAME)
    
    # load data
    with open(DATA_TRAIN_INDEX_PATH, "r", encoding="utf-8") as f:
        print("Loading Data...")
        train_files = []
        for line in f:
            train_files.append(json.loads(line))
    
    with open(DATA_EVAL_INDEX_PATH, "r", encoding="utf-8") as f:
        print("Loading Data...")
        eval_files = []
        for line in f:
            eval_files.append(json.loads(line))

    if len(eval_files) == 0:
        train_files, eval_files = split_data(train_files)
       
    train_batch_nums = int(len(train_files) / batch_size)
    eval_batch_nums = int(len(eval_files) / batch_size)

    random.shuffle(train_files)
    random.shuffle(eval_files)

    train_files = train_files[:train_batch_nums*batch_size]
    eval_files = eval_files[:eval_batch_nums*batch_size]

    train_set = NotaGenDataset(train_files)
    eval_set = NotaGenDataset(eval_files)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=local_rank)

    train_set = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (train_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if not LOAD_FROM_CHECKPOINT:
        if os.path.exists(PRETRAINED_PATH):
            # Load pre-trained checkpoint weights (main checkpoint)
            print("Loading pretrained weights")
            
            # Tie lm_head.weight before loading to avoid mismatch
            if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                model.lm_head.weight = model.transformer.wte.weight.clone().detach()
            
            if torch.cuda.device_count() > 1 and hasattr(model, "module"):
                cpu_model = deepcopy(model.module)
                # Load composite state dict into cpu_model
                load_composite_state_dict(cpu_model, PRETRAINED_PATH, CONTROL_ENCODER_PATH, from_torch=True)
                model.module.load_state_dict(cpu_model.state_dict(), strict=False)
            else:
                cpu_model = deepcopy(model)
                load_composite_state_dict(cpu_model, PRETRAINED_PATH, CONTROL_ENCODER_PATH, from_torch=True)
                model.load_state_dict(cpu_model.state_dict(), strict=False)

            # Re-tie weights after loading
            if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                model.lm_head.weight = model.transformer.wte.weight

            print(f"Successfully Loaded Pretrained Checkpoint")
            pre_epoch = 0
            best_epoch = 0
            min_eval_loss = 100
        else:
            raise Exception('Pre-trained Checkpoint not found. Please check your pre-trained ckpt path.')

    else:
        if os.path.exists(WEIGHTS_PATH):
            print("Loading checkpoint for resuming training")
            checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

            # Tie lm_head.weight before loading to avoid mismatch
            if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                model.lm_head.weight = model.transformer.wte.weight.clone().detach()

            if torch.cuda.device_count() > 1 and hasattr(model, "module"):
                cpu_model = deepcopy(model.module)
                load_composite_state_dict(cpu_model, PRETRAINED_PATH, CONTROL_ENCODER_PATH,from_torch=True)
                model.module.load_state_dict(cpu_model.state_dict(), strict=False)
            else:
                cpu_model = deepcopy(model)
                load_composite_state_dict(cpu_model, PRETRAINED_PATH, CONTROL_ENCODER_PATH,from_torch=True)
                model.load_state_dict(cpu_model.state_dict(), strict=False)

            # Re-tie weights after loading
            if hasattr(model, "lm_head") and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                model.lm_head.weight = model.transformer.wte.weight

            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_sched'])
            pre_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            min_eval_loss = checkpoint['min_eval_loss']

            print(f"Successfully Loaded Checkpoint from Epoch {pre_epoch}")
            checkpoint = None
    
    no_improvement_counter = 0 

    for epoch in range(1+pre_epoch, NUM_EPOCHS+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_loss = train_epoch(epoch)
        eval_loss = eval_epoch()
        wandb.log({"eval_loss": eval_loss})
        if global_rank==0:
            with open(LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_loss < min_eval_loss:
                no_improvement_counter = 0
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = { 
                                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sched': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                'min_eval_loss': min_eval_loss
                                }
                torch.save(checkpoint, WEIGHTS_PATH)
                if(HUGGINGFACE_PATH):
                    if(hasattr(model, "module")):
                        model.module.push_to_hub(HUGGINGFACE_PATH)
                    else:
                        model.push_to_hub(HUGGINGFACE_PATH)
            else:
                no_improvement_counter+=1
                if(no_improvement_counter>EARLY_STOPPING_EPOCHS):
                    print("Stopping early due to model degredation")
                    break
        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Min Eval Loss : "+str(min_eval_loss))