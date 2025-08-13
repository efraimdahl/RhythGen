V_CONTROL = None #Set to "V:1" (or any wanted voice) to mask out other voices during training.  

COND_MODE = "x-attn"  #in-attn; in-attention, good for categorical labels
                      #x-attn: attention-modulation, good for continuous feature vectors.
                      #None: No control vectors are incorperated

COND_FORMAT = "con" #"con" #cat = categorical, con = continuous. 

GATE_INIT = 10.0 #How strong are controls initialized for x-attn

#Small transformer for learning rich embeddings of controls.
ENCODE_CONTROLS = False #Plug in a small transformer before
ENCODER_LAYERS = 1
ENCODER_HEADS = 4

COND_CHAR = False #Add conditioning to character level - decoder, not reccomended

#MODEL CONFIGURATION
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                # Patch Size
PATCH_LENGTH = 2048                                             # Patch Length
CHAR_NUM_LAYERS = 3                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 12                                          # Number of layers in the encoder
HIDDEN_SIZE = 768                                               # Hidden Size


