CONTROL_TERM = "ima" #Term to denote the feature, set as wanted, but match the training data or inference prompt.


#For Categorical Conditioning
N_CLASSES = 7
PADDING_INDEX = 0 #Null value in conditioning

#For continous conditioning
GRID_RESOLUTION=48 #Min value of represented notes default 96, i.e smallest note value possible is 96th notes
MAX_RATIO = 1 #Maximum value of meter ratio represented (i.e how many whole notes are allowed at maximum, i.e 1 allows for meters like 4/4, 2/4 but not for 5/4)
GRID_SIZE = int(GRID_RESOLUTION*MAX_RATIO)

ONSET_FILE_PATH = "preprocessing/labels/baseline_onsets.json"

BEAT_STRENGTH_PATH = f"preprocessing/proto_beat_str_{GRID_RESOLUTION}.json"

USE_AUGMENTED=True                                              # Whether or not to use key augmented data.
EVAL_SPLIT = 0.1                                                # Evaluation Split
MAX_BAR_NUM = 64                                                # Maximum Piece Length
MIN_BAR_NUM = 16                                                # Minimum Piece Length