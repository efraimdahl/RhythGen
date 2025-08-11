from music21 import note, meter, stream
from .data_config import GRID_RESOLUTION, BEAT_STRENGTH_PATH
import json
all_meters = ['6/8', '3/4', '2/4', '3/8', '4/4', '9/8', '2/2', '12/8', '6/4', '2/8', '6/16', '4/8', '5/4', '9/4', '8/8', '3/2', '4/2', '7/8', '5/8', '1/4', '4/16', '7/4', '15/8', '9/16']

def get_beat_strength(given_meter, grid_resolution=96):
    beats, beat_unit = map(int, given_meter.split("/"))  # e.g., 4/4 -> beats=4, beat_unit=4

    # Total duration of the bar in units of 1/max_denom notes
    bar_duration = int((beats / beat_unit) * grid_resolution)

    n = note.Note('E-3')
    n.quarterLength =(1/((grid_resolution)/4))

    s = stream.Stream()

    s.insert(0.0, meter.TimeSignature(given_meter))

    s.repeatAppend(n, bar_duration)

    return [n.beatStrength for n in s.notes]

beat_strength_dic = {}
for m in all_meters:
    bs = get_beat_strength(m,grid_resolution=GRID_RESOLUTION)
    beat_strength_dic.update({m:bs})

with open(f"{BEAT_STRENGTH_PATH}","w") as file:
    json.dump(beat_strength_dic,file,indent=2)