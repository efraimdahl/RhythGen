import torch
import random
import bisect
import json
import re
from .model_config import *
from data.data_config import *
from transformers import GPT2Model,PreTrainedModel,GenerationMixin, GPT2PreTrainedModel, PretrainedConfig, GPT2LMHeadModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from tokenizers import Tokenizer
from typing import Optional, Tuple, Union, Callable

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Block, GPT2Attention, get_device_map, assert_device_map


from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa
)
from transformers.pytorch_utils import Conv1D
from transformers.utils import logging
from safetensors.torch import load_file

logger = logging.get_logger(__name__)


class ControlConfig(PretrainedConfig):
    def __init__(self, control_dim=1, embed_dim=768, nhead=4, nlayer=1, intermediate_dim=256, grid_size=48,ntarget=1, dropout=0.1,**kwargs):
        super().__init__(**kwargs)
        self.control_dim = control_dim
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.intermediate_dim = intermediate_dim
        self.grid_size = grid_size
        self.ntarget = ntarget
        self.nlayer=nlayer
        self.dropout=dropout


class AttentionPool(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, hidden_size))  # [1, 1, H]
        self.scale = hidden_size ** -0.5

    def forward(self, x):  # x: [B, G, H]
        B, G, H = x.shape
        #print(x.shape)
        # Repeat the query across the batch
        q = self.query.expand(B, 1, H)  # [B, 1, H]
        k = x  # [B, G, H]
        v = x  # [B, G, H]

        # Attention scores: [B, 1, G]
        attn_scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # [B, 1, G]

        # Weighted sum: [B, 1, H] → squeeze to [B, H]
        pooled = torch.matmul(attn_weights, v).squeeze(1)  # [B, H]
        pooled=pooled.unsqueeze(0)  # [1, B, H]
        return pooled



class ControlTransformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.metric_proj = torch.nn.Linear(config.control_dim, config.embed_dim)
        self.spectral_proj = torch.nn.Linear(config.control_dim, config.embed_dim)

        self.pos_emb = torch.nn.Parameter(torch.randn(config.grid_size, config.embed_dim))
        self.input_norm = torch.nn.LayerNorm(config.embed_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.nhead, dim_feedforward=config.intermediate_dim,dropout=config.dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=config.nlayer)

    def forward(self, controls, metric_weights=None, mask=None):
        B, seq_len, _ = controls.shape
        
        x = self.spectral_proj(controls)  # [B, 48, embed_dim]
        #if(metric_weights):
        #    x_m = self.metric_proj(metric_weights)  # [B, 48, embed_dim]
        #    x = x_s + x_m  # [B, 48, embed_dim]
        #else:
        #    x=x_s
        x = x + self.pos_emb.unsqueeze(0)              # add positional encoding
        assert x.shape[1] == self.pos_emb.shape[0], "Mismatch in sequence length and positional embedding"
        x = x.transpose(0, 1)                           # [48, B, embed_dim]
        x = self.encoder(x)                             # [48, B, embed_dim]
        x = x.transpose(0, 1)                           # [B, 48, embed_dim]
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            x = x * mask  # zero out full sequence for dummy examples

        return x
    
class PatchLevelDecoder(PreTrainedModel):
    """
    A Patch-level Decoder model for generating patch features in an auto-regressive manner. 
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model_Con(config)
        if COND_MODE == 'in-attn' and COND_FORMAT=="cat":
            self.control_embedding = torch.nn.Embedding(N_CLASSES, config.n_embd,padding_idx=PADDING_INDEX)
        
        elif COND_MODE== 'in-attn' and COND_FORMAT=="con" and not ENCODE_CONTROLS:
            self.control_projection = torch.nn.Linear(GRID_SIZE, config.n_embd)
        elif COND_MODE== 'in-attn' and COND_FORMAT=="con" and ENCODE_CONTROLS:
            self.control_projection = AttentionPool(config.n_embd)

    def forward(self,
                patches: torch.Tensor,
                controls: torch.Tensor,
                masks=None,
                control_scheduler:float=1) -> torch.Tensor:
        """
        The forward pass of the patch-level decoder model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=128).to(self.dtype)
        patches = patches.reshape(len(patches), -1, PATCH_SIZE * (128))
        patches = self.patch_embedding(patches.to(self.device))

        control_emb = None
        
        if(COND_FORMAT=="cat" and COND_MODE == 'in-attn'):
            control_emb = self.control_embedding(controls.to(self.device))
        elif(COND_FORMAT=="con" and COND_MODE == 'in-attn'):

            control_emb = self.control_projection(controls.to(self.device))

        if(COND_MODE):
            controls = control_scheduler*controls
            if(control_emb is not None):
                control_emb = control_scheduler*control_emb
        #print("in sequence generation: Patches:",patches.shape, "  Controls:",controls.shape)

        if masks==None:
            return self.base(inputs_embeds=patches, control_embeds=control_emb, controls=controls)
        else:
            return self.base(inputs_embeds=patches,
                             control_embeds=control_emb,
                             attention_mask=masks,
                             controls=controls)


class CharLevelDecoder(PreTrainedModel):
    """
    A Char-level Decoder model for generating the chars within each patch in an auto-regressive manner
    based on the encoded patch features. It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.max_position_embeddings = config.max_position_embeddings
        self.base = GPT2LMHeadModel(config)
    """
    def get_control_embeddings(self, controls, generation=False):
        control_emb = None
        if(COND_FORMAT=="cat" and COND_MODE == 'in-attn' and COND_CHAR):
            
        
            # Expand embedding across sub-dimension (repeat across PE)
            if(not generation):
                controls =  controls[:, :-1]  # [1, S-1]
                control_emb = self.control_embedding(controls.to(self.device))
                control_emb = control_emb.squeeze(0)  # # [S, D]
                control_emb = control_emb.unsqueeze(1).expand(-1, self.max_position_embeddings, -1)  # [S, PE, D]
            
        elif(COND_FORMAT=="con" and COND_MODE == 'in-attn' and COND_CHAR):
            control_emb = self.control_embedding(controls.to(self.device))
            control_emb = self.control_projection(controls.to(self.device))
        return(control_emb)
    """
    
    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the char-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        """
        # preparing the labels for model training
        target_patches = torch.cat((torch.ones_like(target_patches[:,0:1])*self.bos_token_id, target_patches), dim=1)
        # print('target_patches shape:', target_patches.shape)

        target_masks = target_patches == self.special_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        # select patches
        if PATCH_SAMPLING_BATCH_SIZE!=0 and PATCH_SAMPLING_BATCH_SIZE<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:PATCH_SAMPLING_BATCH_SIZE])

            target_patches = target_patches[selected_indices,:]
            target_masks = target_masks[selected_indices,:]
            encoded_patches = encoded_patches[selected_indices,:]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)
        
        #control_emb = self.get_control_embeddings(controls)


        output = self.base(inputs_embeds=inputs_embeds,
                            attention_mask=target_masks,
                            labels=labels)
                         # output_hidden_states=True=True)

        return output

    def generate_pre_logits(self,
                 encoded_patch: torch.Tensor, 
                 tokens: torch.Tensor): # [1]
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability logitst for the next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1) # [1, 1, hidden_size]
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        
       
        # Get output from model
        outputs = self.base(inputs_embeds=tokens)
        
        return outputs.logits.squeeze(0)[-1]



    def generate(self,
                 encoded_patch: torch.Tensor,   # [hidden_size]
                 controls:torch.Tensor,
                 tokens: torch.Tensor): # [1]
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1) # [1, 1, hidden_size]
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        
        # Get output from model
        outputs = self.base(inputs_embeds=tokens,
        )
        
        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs

class RhythGenModel(PreTrainedModel):
    """
    RhythGen is a language model with a hierarchical structure based on NotaGen.
    It includes a patch-level decoder and a char-level decoder.
    The patch-level decoder is used to generate patch features in an auto-regressive manner.
    The char-level decoder is used to generate the chars within each patch in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, encoder_config, decoder_config, control_config=None):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)
        if(ENCODE_CONTROLS):
            if(control_config==None):
                raise ValueError("Please provide a configuration for the control transformer")            
            self.control_encoder = ControlTransformer(control_config)
        

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor,
                controls: torch.Tensor,
                epoch:int=None,
                ):
        """
        The forward pass of the bGPT model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the decoded patches
        """
        
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        #print("Patch Shape", patches.shape, "  Control Shape",controls.shape)
        control_mask=None
        if(V_CONTROL):
            control_mask  = (controls.abs().sum(dim=-1, keepdim=True) > 0).float().squeeze(0)  # [1, sequence, 1]
            
        if(ENCODE_CONTROLS):
            B, num_patches, seq_len = controls.shape
            controls = controls.view(B * num_patches, seq_len, 1)
            controls = self.control_encoder(controls)  # [B, grid_size,hidden_size]
            B, grid, hidden_size = controls.shape
            if control_mask is not None:
                #print("mask shape before", control_mask.shape, controls.shape)
                control_mask = control_mask.unsqueeze(-1).expand(-1, grid, hidden_size)  #
                #print("Control Code Shape",control_mask.shape, controls.shape)
                controls= controls*control_mask
        control_scheduler = 1.0
        if(ANNEALING_EPOCHS and epoch):
            control_scheduler = min(1,((epoch/ANNEALING_EPOCHS)-(1/ANNEALING_EPOCHS)))
            
        encoded_patches = self.patch_level_decoder(patches, 
          controls, 
          masks, 
          control_scheduler=control_scheduler)["last_hidden_state"]
        
        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0
        
        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]        

        return self.char_level_decoder(encoded_patches, 
          #controls=controls, 
          target_patches=patches)
        
    
    def generate_cfg(self,
                 patches: torch.Tensor,
                 controls: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0,
                 guidance_scale=1,
                 vpatch=False,
                 current_control=0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param controls: the control conditions for generation
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :param guidance_scale: classifier free guidance scale for generation
        :return: the generated patches
        """

        if patches.shape[-1] % PATCH_SIZE != 0:
            tokens = patches[:,:,-(patches.shape[-1]%PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%PATCH_SIZE)]
        else:
            tokens =  torch.tensor([self.bos_token_id], device=self.device)
        
        if(not vpatch):
            controls[0][-1] = torch.full_like(controls[0][-1], fill_value=PADDING_INDEX)
        
        #print("Control Val", controls[0][-1])

        if(ENCODE_CONTROLS):
            B, num_patches, seq_len = controls.shape
            controls = controls.view(B * num_patches, seq_len, 1)
            controls = self.control_encoder(controls)  # [B, grid_size,hidden_size]
        
        patches = patches.reshape(len(patches), -1, PATCH_SIZE) # [bs, seq, patch_size]
        #print("Patch Shape", patches.shape, "  Control Shape",controls.shape)

        ncond_controls = torch.full_like(controls, fill_value=PADDING_INDEX)

        cond_encoded_patches = self.patch_level_decoder(patches,controls)["last_hidden_state"]    # [bs, seq, hidden_size]
        uncond_encoded_patches = self.patch_level_decoder(patches,ncond_controls)["last_hidden_state"]    # [bs, seq, hidden_size] 
        #print("Encoded Patch Uncond", uncond_encoded_patches.shape, "  Encoded Patch Cond", cond_encoded_patches.shape)

        generated_patch = []         
        generated_chars = ""
        current_voice = None
        while True:
            
            temp_guidance_scale = guidance_scale if vpatch and (V_CONTROL!=None) else 0 
            #print("Shape of encoded patches", cond_encoded_patches.shape, cond_encoded_patches[0][-1].shape, controls[:,-1].shape)
            cond_outputs = self.char_level_decoder.generate_pre_logits(cond_encoded_patches[0][-1], tokens=tokens)# [128]
            uncond_outputs = self.char_level_decoder.generate_pre_logits(uncond_encoded_patches[0][-1], tokens=tokens)
            
            guided_logits = uncond_outputs + guidance_scale * (cond_outputs - uncond_outputs)
            prob = torch.nn.functional.softmax(guided_logits, dim=-1).cpu().detach().numpy()
            #print("lm_head weight shape:", self.char_level_decoder.base.lm_head.weight.shape)  # should be [128, hidden_dim]
            #print("Logits shape before sampling:", prob.shape, guided_logits.shape, uncond_outputs.shape, cond_outputs.shape)  # should be [128]
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True) # [128]
            
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True) # [128]
            

            token = temperature_sampling(prob, temperature=temperature) # int
            char = chr(token)
            generated_chars+=char
            voice_tags = list(re.finditer(r'\[V:([^\]]+)\]', generated_chars))
            #print(generated_chars,voice_tags)

            if V_CONTROL and voice_tags:
                last_voice = voice_tags[-1].group(0)
                if(last_voice!=current_voice):
                    vpatch = (last_voice == V_CONTROL)
                    if(not vpatch):
                        controls[0][-1] = torch.full_like(controls[0][-1], fill_value=PADDING_INDEX)
                    else:
                        controls[0][-1] = torch.full_like(controls[0][-1], fill_value=current_control)
                    cond_encoded_patches = self.patch_level_decoder(patches,controls)["last_hidden_state"]
                    current_voice = last_voice

            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:# or token == self.eos_token_id:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch,vpatch


    
    def generate(self,
                 patches: torch.Tensor,
                 controls: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """

        if patches.shape[-1] % PATCH_SIZE != 0:
            tokens = patches[:,:,-(patches.shape[-1]%PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%PATCH_SIZE)]
        else:
            tokens =  torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, PATCH_SIZE) # [bs, seq, patch_size]
        #print("Patch Shape", patches.shape, "  Control Shape",controls.shape)

        encoded_patches = self.patch_level_decoder(patches,controls)["last_hidden_state"]    # [bs, seq, hidden_size]
        generated_patch = []         

           

        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()  # [128]
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True) # [128]
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True) # [128]
            token = temperature_sampling(prob, temperature=temperature) # int
            char = chr(token)
            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:# or token == self.eos_token_id:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch



def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights

class GPT2Attention_Con(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, control_dim = GRID_SIZE):
        super().__init__(config)
        
        if(COND_MODE=="x-attn"): # Attention dimension [B, grid_size,hidden_size]
            if(ENCODE_CONTROLS):
                self.q_control_projection = AttentionPool(config.n_embd)
                self.k_control_projection = AttentionPool(config.n_embd)
                self.v_control_projection = AttentionPool(config.n_embd)

            else:
                # Control projections
                self.control_q_proj = torch.nn.Linear(control_dim, self.embed_dim)
                self.control_k_proj = torch.nn.Linear(control_dim, self.embed_dim)
                self.control_v_proj = torch.nn.Linear(control_dim, self.embed_dim)

            self.q_gate_scale = torch.nn.Parameter(torch.full((config.n_embd,), GATE_INIT))  
            self.k_gate_scale = torch.nn.Parameter(torch.full((config.n_embd,), GATE_INIT)) 
            self.v_gate_scale = torch.nn.Parameter(torch.full((config.n_embd,), GATE_INIT))

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        controls: Optional[torch.FloatTensor]=None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        
        if(COND_MODE=="x-attn"):
            #print(controls.sum())
            if(ENCODE_CONTROLS==True):
                control_q = self.q_control_projection(controls)
                control_k = self.k_control_projection(controls)
                control_v = self.v_control_projection(controls)
            else:
                # Project control (shape: [B, T, control_dim]) — preprocessed externally
                control_q = self.control_q_proj(controls)
                control_k = self.control_k_proj(controls)
                control_v = self.control_v_proj(controls)        

            # Apply control with gating
            query_states = query_states + self.q_gate_scale * control_q
            key_states = key_states + self.k_gate_scale * control_k
            value_states = value_states + self.v_gate_scale * control_v


        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        


        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None

        is_cross_attention = encoder_hidden_states is not None
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentionned options are provided).
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2Block_Con(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config)

        self.attn = GPT2Attention_Con(config=config, layer_idx=layer_idx)
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        controls: Optional[torch.FloatTensor]=None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            controls=controls,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                controls=controls
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2Model_Con(GPT2Model):
    _supports_param_buffer_assignment = False

    def __init__(self, config):
        super().__init__(config)
        self.h = torch.nn.ModuleList([GPT2Block_Con(config, layer_idx=i) for i in range(config.num_hidden_layers)])
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        control_embeds: Optional[torch.LongTensor]=None,
        controls: Optional[torch.FloatTensor]=None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if control_embeds is not None and COND_MODE=="in-attn":
                #print(hidden_states.shape,control_embeds.shape)

                hidden_states = hidden_states + control_embeds

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                    controls,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    controls=controls,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallelt: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
