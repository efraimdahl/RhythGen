## Data Pre-processing

### Batch Conversion
For your convenience we have set up a bunch of batch conversion scripts to prepare your data for training and evaluation. 

For the conversion from midi to xml, mscx2xml and xml2midi you will need to 
download and install the musescore commandline software. 

To convert in bulk, set the ```ORI_FOLDER``` abd ```DES_FOLDER``` variables in either 
Then navigate to the `preprocessing/scripts` folder and run.

```
python batch_midi2xml.py
```

Use the ```musescore_options.xml``` to set export options (buggy).  

### Extract Onsets

In `data_config.py` you can make some important decisions, i.e decide on the size and range of feature vectors, the maximum resolution of the piece, and whether data augmentation is applied.

For labelling the data the onsets are extracted from the piece. The ABCRhythmTool from the  `abc_rhythm` file is applied on each piece to extracts note density, syncopation scores, as well as inner metric analysis from the onsets parsed from the ABC files. You will need to provide a proto_rhythms file, which contains the beat_strength profile for every meter. This is calculated in `get_beat_str.py` which you can run from the root directory by running.

```
python -m preprocessing.get_beat_str_profile 
```

You can test the labelling on a single file by running 
```
python -m preprocessing.abc_rhythm
```

The `1_extract_onsets_abc.py` uses ABCRhythmTool to extract the scores, inner metric/spectral weights and onsets from each piece, additionally it performs quantile based labelling based on the note-density and syncopation scores.



Modify the ```ORI_FOLDER```, ````ONSET_FILE_LOCATION```, `ONSET_FILE_NAME`. Make sure to correctly indicate whether your ABC files are in interleaved or regular abc notation by setting `INTERLEAVED`. If you want to avoid statistical labeling, you can set manual boundaries using `MANUAL_BOUNDARIES`, just make sure this matches your `N_CLASSES` from `data_config.py`. Set the `TARGET_ONSET_FILE` if you want to use quantile boundaries from another (already processed) dataset. 

Then run

```
python -m preprocessing.1_extract_onsets_abc
```

### Prepare the training set


Modify the ```ORI_FOLDER```, ```INTERLEAVED_FOLDER```, ```AUGMENTED_FOLDER```, and ```FEATURE_TYPE``` in ```2_prepare_training_set.py```:
  
```python

ORI_FOLDER = "data/example/LB/abc"  # Replace with the path to your folder containing standard ABC notation files
INTERLEAVED_FOLDER = "data/example/LB_training_sync/interleaved"   # Output interleaved ABC notation files to this folder
AUGMENTED_FOLDER = 'data/example/LB_training_sync/augmented'   # Output key-augmented and rest-omitted ABC notation files to this folder

ONSET_DATA = 'data/labels/LB_onsets.json' #Point this to your extracted onset file.

PATH_REPLACEMENT_TERM = "abc_test" #"abc_test" #if onset key differes from file location. (default is abc)

```
then run this script from the root directory:
```
python -m preprocessing.2_prepare_training_set
```
- The script will convert the standard ABC to interleaved ABC, which is compatible with CLaMP 2. The files will be under ```INTERLEAVED_FOLDER```.

- This script will make 15 key signature folders under the ```AUGMENTED_FOLDER```, and output interleaved ABC notation files with rest bars omitted. This is the data representation that NotaGen adopts.

- This script will also generate data index files for training NotaGen. It will randomly split train and eval sets according to the proportion ```EVAL_SPLIT``` defines. The index files will be named as ```{AUGMENTED_FOLDER}_train.jsonl``` and ```{AUGMENTED_FOLDER}_eval.jsonl```.

## Data Post-processing

### Preview Sheets in ABC Notation

We recommend [EasyABC](https://sourceforge.net/projects/easyabc/), a nice software for ABC Notation previewing, composing and editing.

It may be neccessary to add a line "X:1" before each piece to present the score image in EasyABC.

### Check and Repair bar alignment. 
Use the file `3_correct_alignment.py` which checks and fixes bar alignment in `.abc` files from `FILE_PATH`, saves corrected files to `TARGET_PATH`, and prints alignment stats.

1. Set `FILE_PATH` and `TARGET_PATH` in the script.
2. Run from the root directory of the project:
```bash
   python preprocessing.3_correct_alignment
```

### Convert to MusicXML
Use the batch conversion scripts as shown above.
