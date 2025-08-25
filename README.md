# 🎵 RhythGen: Controlling Rhythmic Complexity in generated symbolic music for application in serious games

## 📖 Overview
**RhythGen** is a fine-tuned symbolic music generation model based on [NotaGen](https://github.com/ElectricAlexis/NotaGen), small, with rhythmic conditioning on syncopation levels and note density. We explore the effect of different conditioning mechanisms, conditioning attributes, and data preparation models on quality and control adherence. 

<p align="center">
  <!-- HuggingFace -->
  <a href="https://huggingface.co/efraimdahl/RhythGen">
    <img src="https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=" alt="Weights">
  </a>
  &nbsp;&nbsp;
  <!-- Collab -->
  <a href="https://colab.research.google.com/drive/16TRtFGECxh6nsh7WvUktDS3hvGoPhtva?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

## ⚙️ Environment Setup

```bash
conda create --name rhythgen python=3.10
conda activate rhythgen
conda install pytorch==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install accelerate
pip install optimum
pip install -r requirements.txt
```

## 🏋️ RhythGen Model Weights

### Pre-training
We use [NotaGen-small](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth) with 110M parameters, 12 patch-level decoder layers, 3 character-level decoder layers, a hidden size of 768, and a patch length (context length) of 2048. 

### Fine-tuning

We fine-tuned NotaGen-small on a corpus of approximately 1000 pieces from either the [Lieder Dataset](https://github.com/OpenScore/Lieder), which is public, or the [RAG Collection](https://dspace.library.uu.nl/bitstream/handle/1874/354841/OdekerkenVolkKoops2017.pdf?sequence=1), which is available on request. 

The finetuned weights are available here: https://huggingface.co/efraimdahl/RhythGen.

#### Descriptors
We use one of four descriptors to condition the generator. All descriptors are calculated from the onsets of the music extracted with the `ABCRhythmTool` in the `abc_rhythm.py` file, which also quantifies the onsets to a metric grid of a size depending on the parameter `min_note_val`, which determines the smallest possible rhythmic unit. Syncopation labels, spectral and metric weights are extracted from the onsets using [pyinmean](https://github.com/efraimdahl/pyinmean)

- **Note Density Labels**
  - Based on the average distance between two consecutive onsets in one bar, 
  - binned into one of six discrete labels based on quantiles of average onset distance across the dataset.
  - One value per bar
  
- **Syncopation Labels**
  - Based on a syncopation measure introduced in [Inner Metric Analysis as a Measure of Rhythmic Syncopation](https://durham-repository.worktribe.com/output/3342386) by Bemman and Christensen
  - Binned into one of six discrete labels based on quantiles across the dataset.
  - One value per bar

- **Spectral Weight Profiles**
  - Assign values to every position on a fixed metric grid by extending local meters.
  - Vector of size resolution*max_bar_length per bar

- **Metric Weight Profiles**
  - Assign values to every onset by quantifying local meters. All other values of the metric grid are set to 0. 
  - Vector of size resolution*max_bar_length per bar


#### Conditioning Mechanisms
We use one of three conditioning mechanisms to condition the model. The mechanisms are implemented alongside the original NotaGen model in `model.py`. Use `config.py` to adjust the mechanism used. In addition to the conditioning mechanisms, we also use classifier-free guidance for all models with a dropout rate of 30% and a varying guidance scale. Some experiments include a condition encoder that extracts features from the metric or spectral weight profiles. We pretrain this condition encoder on a regression task of calculating a syncopation score and/or note-density of a bar, given metric/spectral weight profiles of a bar. The control-encoder seems to improve results somewhat, but not dramatically. The successful configurations isolate syncopated voices, either by training on selected voices or through masked conditioning during training, which is set by the `VCOND` variable. So far, the inference script does not adhere to the voice targeting, which results in higher error rates.

- **Text-Based Conditioning**
  - Default conditioning patchway, add syncopation labels into the ABC files for training, force them into the sequence while generating. Similar to the conditioning mechanism described for fine-tuning in the original [NotaGen](https://arxiv.org/abs/2502.18008) and [Tunesformer](https://arxiv.org/abs/2301.02884) papers. Don't change anything about the model architecture.
  - Works poorly. 
  
- **In-Attention Conditioning**
  - Based on a condition mechanism introduced in [MuseMorphose](https://github.com/YatingMusic/MuseMorphose)
  - Add learned embeddings of the control labels to the hidden state in each self-attention layer of the patch-level decoder.
  - Works for syncopation and note-density labels. 

- **Attention Modulation**
  - Original conditioning mechanism
  - Adds learned projections of the control to the Key, Query, and Value matrices in each self-attention layer of the patch-level decoder.
  - Works for using spectral weights to control note-density, but syncopation is not captured. Metric weights do not yield good control. 


#### Settings
Choose your configuration of the model by setting the variables in the `rhythgen/model_config.py` file. 
```python
V_CONTROL = None #Set to "V:1" (or any wanted voice) to mask out other voices during training.  

COND_MODE = "in-attn" #in-attn; in-attention, good for categorical labels
                      #x-attn: attention-modulation, good for continuous feature vectors.
                      #None: unconditioned

COND_FORMAT = "cat" #"con" #cat = categorical, con = continuous.  The continuous representation uses a projection layer of GRID_SIZE set in the preprocessing/data_config.py file. The embedding layer uses an embedding layer with N_CLASSES different classes set in the preprocessing/data_config.py.

GATE_INIT = 10.0 #How strong are controls initialized for x-attn

#Small transformer for learning rich embeddings of controls.
ENCODE_CONTROLS = False #Plug in a small transformer before
ENCODER_LAYERS = 1
ENCODER_HEADS = 4

COND_CHAR = False #Add conditioning to character level - decoder, not recommended
```

## 📚 Data
In the `data` folder, you can find a collection of music generated by different model configurations described in my [Masters Thesis](). 
If you are performing research and are interested in the results from my listener study, reach out to me here.

## 🎹 Demo

### Online Colab Demo

[Online Colab Demo](https://colab.research.google.com/drive/16TRtFGECxh6nsh7WvUktDS3hvGoPhtva?usp=sharing)


## 🛠️ Data Pre-processing & Post-processing

For converting **ABC notation** files from/to **MusicXML** files, and to label the data with extracted rhythmic features for training, please view [preprocessing/README.md](https://github.com/efraimdahl/RhythGen/blob/main/preprocessing/README.md) for instructions.

## 🎯 Fine-tune

Here we give an example of fine-tuning **RhythGen** with the labeled example data achieved by running the preprocessing scripts on the data found in `data/example/LB/xml`.

### Configuration
- In ```finetune/ft_config.py```:
  - Modify the ```DATA_TRAIN_INDEX_PATH``` and ```DATA_EVAL_INDEX_PATH``` to match your data path from the preprocessing.
  - Download pre-trained NotaGen (small) weights, and modify the ```PRETRAINED_PATH```.
  - ```EXP_TAG``` is for differentiating the models. It will be integrated into the ckpt's name. 
  - You can also modify other parameters like the learning rate.

### Execution
Use this command for fine-tuning from the directory of the repository:
```bash
python -m finetune.finetune
```

## Inference 
For inference, adjust the settings in `inference/if_config.py`
```python
import os
# Configurations for inference LiederLabled_TVDISTPretrained/L.safetensors =
INFERENCE_WEIGHTS_PATH = '../Pretrained/RAS2.safetensors'               # Path to weights for inference# Folder to save output files
NUM_SAMPLES = 150                                              # Number of samples to generate (only for generate mode)

#SAMPLING SETTINGS
TOP_K = 9                                              # Top k for sampling
TOP_P = 0.9                                            # Top p for sampling
TEMPERATURE = 1.2                                      # Temperature for sampling

ORIGINAL_OUTPUT_FOLDER = os.path.join('../output/original', os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))
INTERLEAVED_OUTPUT_FOLDER = os.path.join('../output/interleaved', os.path.splitext(os.path.split(INFERENCE_WEIGHTS_PATH)[-1])[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))

PROMPT_PATH = "../RhythGen/data/example/LB_training_sync/augmented/C" #set to either a directory of abc files, their labels and metadata will be used as prompt, or set to a file where each line contains a set of conditioning labels.

CFG_GUIDANCE=[1,3] #generates NUM_SAMPLES for each item in list, 0 = Unconditioned, 1 = Regular Conditioned >1 = Boosted Conditioned
STARTING_CONDITION = (0,0) #First index number, second index guidance scale. 
```
Then, from the root directory of the repository, run
```bash
python -m inference.inference
```
### ⚙️ Evaluation Setup

Download model weights and put them under the ```clamp2/``` folder:
- [CLaMP 2 Model Weights](https://huggingface.co/sander-wood/clamp2/blob/main/weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth)
- [M3 Model Weights](https://huggingface.co/sander-wood/clamp2/blob/main/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth)

### 🔍 Extract Ground Truth Features
Modify ```input_dir``` and ```output_dir``` in ```clamp2/extract_clamp2.py``` to match your interleaved abc folder, and set an output folder as desired:
```python
input_dir = '../data/[DATSET_NAME]'  # interleaved abc folder
output_dir = 'feature/[DATSET_NAME]'  # feature folder
```
Extract the features:
```
cd clamp2/
python extract_clamp2.py
```

#### 3. Statistics on Average CLaMP 2 Score
If you're interested in the **Average CLaMP 2 Score** of the current model, modify the parameters in ```clamp2/statistics.py```:
```python
gt_feature_folder ='feature/[DATSET_NAME]'
output_feature_folder = 'feature/[YOUR_MODEL_NAME]'
```
Then run this script:
```
cd clamp2/
python statistics.py
```

## 📚 Citation

Coming Soon
