
#DATA CONFIGURATION
DATA_TRAIN_INDEX_PATH = "data/example/LB_training/augmented_train.jsonl" 
DATA_EVAL_INDEX_PATH  = "data/example/LB_training/augmented_eval.jsonl"

#EXPERIMENT NAME
PUSH_TO_HF=False                                              #Save to huggingface. 
HUGGINGFACE_PATH= ""
EXP_TAG = ""                                            # Experiment tag for name differentiation

#CONTROL CONFIGURATION
FREEZE = False #Freeze original layers.
P_CONTROL_DROPOUT = 0.3 #Probability of control dropout for an entire sequence, important for CFG
ANNEALING_EPOCHS = None #Set to None, or an integer denoting the number of epochs over which controls are slowly introduced to the model.
EARLY_STOPPING_EPOCHS = 4                                       #No improvement over x epochs will trigger a stop. 

#TRAINING CONFIGURATION
BATCH_SIZE = 1         
LEARNING_RATE = 1e-5   
NUM_EPOCHS = 64                                                 # Number of epochs to train for (if early stopping doesn't intervene)
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full conaudio
LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint
WANDB_LOGGING = False                                           # Whether to log to wandb
WANDB_KEY = ''

PRETRAINED_PATH = "../Pretrained/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth"                # Path of pretrained weights
CONTROL_ENCODER_PATH = None


from rhythgen.model_config import *

NAME =  EXP_TAG + \
        "_p_size_" + str(PATCH_SIZE) + \
        "_p_length_" + str(PATCH_LENGTH) + \
        "_p_layers_" + str(PATCH_NUM_LAYERS) + \
        "_c_layers_" + str(CHAR_NUM_LAYERS) + \
        "_h_size_" + str(HIDDEN_SIZE) + \
        "_lr_" + str(LEARNING_RATE) + \
        "_batch_" + str(BATCH_SIZE)

WEIGHTS_PATH = "weights_notagen_" + NAME + ".pth"                  # Path to save weights
LOGS_PATH    = "logs_notagen_"    + NAME + ".txt"                     # Path to save logs
WANDB_NAME = NAME
