import os
# Configurations for inference LiederLabled_TVDISTPretrained/L.safetensors =
INFERENCE_WEIGHTS_PATH = '../../Pretrained/RagtimeSync_vcond.safetensors'               # Path to weights for inference# Folder to save output files
NUM_SAMPLES = 150                                              # Number of samples to generate (only for generate mode)

#SAMPLING SETTINGS
TOP_K = 9                                              # Top k for sampling
TOP_P = 0.9                                            # Top p for sampling
TEMPERATURE = 1.2                                      # Temperature for sampling

ORIGINAL_OUTPUT_FOLDER = os.path.join('../output/original', os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))
INTERLEAVED_OUTPUT_FOLDER = os.path.join('../output/interleaved', os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))

ROMPT_PATH = "/data/example/data/examle/LB_training/augmented" #set to either a directory of abc files, their labels and metadata will be used as prompt, or set to file where each line contains a set of conditioning labels.

CFG_GUIDANCE=[1,3] #generates for each item in list, 0 = Unconditioned, 1 = Regular Conditioned >1 = Boosted Conditioned
STARTING_CONDITION = (0,0) #First index number, second index guidance scale. 
