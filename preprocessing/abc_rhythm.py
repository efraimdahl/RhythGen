import re
import sys
import os

from .data_config import BEAT_STRENGTH_PATH
from .utils import (
    remove_information_field, 
    remove_bar_no_annotations, 
    Quote_re, 
    Barlines,
    merge_barline_and_bartext_dict,
    extract_metadata_and_parts, 
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict,
    extract_barline_bartext_lyrics_dict,
    extract_barline_and_bartext_dict_rotated)

from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated

import operator
import math
import pyinmean
import json
import numpy as np
from sklearn.preprocessing import normalize
import fractions


with open(BEAT_STRENGTH_PATH) as file:
    proto_rhythms = json.load(file)



def chi2(x, y):
    denom = x + y
    # Avoid division by zero by masking
    mask = denom > 0
    return np.sum(((x[mask] - y[mask]) ** 2) / denom[mask])


def unit_scale(weights):
    """Normalize the weights to sum to 1"""
    return weights / np.sum(weights)


class SyncopationScore():
    def __init__(self,proto_rhythms=None):
        self.repeat_count = 12
        self.proto_rhythms = proto_rhythms
    
    def get_beat_strength_rhythms(self,beat,beat_unit,len_bar):
        if(beat_unit==4 and len_bar==4):
            s = [1,0.25,0.5,0.25]
        elif(beat_unit==4 and len_bar==8):
            s = [1,0.125,0.25,0.125,0.5,0.125,0.25,0.125]
        elif(beat_unit==4 and len_bar==12):
            s = [1.0, 0.0625, 0.0625, 0.25, 0.0625, 0.0625, 0.5, 0.0625,0.0625, 0.25, 0.0625, 0.062]
        elif(beat_unit==8 and len_bar==6):
            s = [1.0, 0.25, 0.25, 0.5, 0.25, 0.25]
        return s
    def retrieve_nominal_rhythm(self, beats, beat_unit, len_bar):
        if(self.proto_rhythms!=None):
            #print("getting proto rhythms")
            meter = f"{beats}/{beat_unit}"
            s = self.proto_rhythms.get(meter)
            if(s==None):
                raise ValueError(f"No corresponding proto rhythm for {meter}")
        else:
            s=self.get_beat_strength_rhythms(beats, beat_unit, len_bar)
        return np.array(s)
    
    def calculate_onsets(self, bar, bar_duration, repeats=1):
        step = int(bar_duration // len(bar))  # duration of each subdivision
        # Compute onsets based on where 1s occur
        onsets = []
        for rep in range(0,repeats):
            offset = rep*bar_duration
            for i, v in enumerate(bar):
                if v == 1:
                    onsets.append(offset+(i * step))
        return onsets

    def extend_onsets(self, onsets, bar_duration, repeats=1):
        # Compute onsets based
        final_onsets = []
        for rep in range(0,repeats):
            offset = rep*bar_duration
            for onset in onsets:
                final_onsets.append(int(onset+offset))
        return final_onsets


    def fold_ima(self, weights, bar_duration):
        
        folded_profile = np.zeros(bar_duration)
        for i in range(0,len(weights)):
            #print(i%bar_duration,i,folded_profile[i%bar_duration],weights[i])
            folded_profile[i%bar_duration] += weights[i]
       
        return folded_profile
    
    def extend_metric_weight(self,weights,onsets):
        ret_weigths = []
        onsetIdx=0
        for i in range(0,int(onsets[-1])+1):
            #print(i,onsetIdx,onsets[onsetIdx])
            #i=int(i)
            if(int(onsets[onsetIdx])==i):
                ret_weigths.append(weights[onsetIdx])
                onsetIdx+=1
            else:
                ret_weigths.append(0.0)
    
        return ret_weigths
    
    def get_bar_duration(self, meter, resolution=None):
        beats, beat_unit = map(int, meter.split("/"))  # e.g., 4/4 -> beats=4, beat_unit=4
        
        tmp_resolution = beat_unit if resolution==None else resolution
        bar_duration = int((beats / beat_unit) * tmp_resolution)
        return bar_duration
    
    def measure(self,folded_ima,s,qi=1,a=1):
        distance = chi2(folded_ima,s)

        if(qi!=None):
            distance = distance**a/len(folded_ima)
        else:
            distance = distance**a
        return distance
    
    def calculate_syncopation_score_bar(self, bar, meter, spectral=True, a=1, qi=1, resolution=None, bar_type = "onsets"):
        
        beats, beat_unit = map(int, meter.split("/"))  # e.g., 4/4 -> beats=4, beat_unit=4
        tmp_resolution = resolution if resolution!=None else max(len(bar),beat_unit)
        bar_duration = int((beats / beat_unit) * tmp_resolution)
        
        if(bar_type=="binary"):
            repeated_onsets = self.calculate_onsets_from_binary(bar, bar_duration, repeats=self.repeat_count)
        elif(bar_type=="onsets"):
            repeated_onsets = self.extend_onsets(bar, bar_duration, repeats=self.repeat_count)
        
        if(spectral and 0 not in repeated_onsets):
            repeated_onsets=[0]+repeated_onsets
           
        ima_res = np.array(pyinmean.get_normalized_ima(repeated_onsets,spectral))
        #print("preshape:",ima_res.shape)
        if(spectral==False):
            ima_res = np.array(self.extend_metric_weight(ima_res,repeated_onsets))
        
        folded_ima = self.fold_ima(ima_res, bar_duration)

        
        if(qi==None):
            s = self.retrieve_nominal_rhythm(beats,beat_unit,len(bar))
            s = unit_scale(s)
        else:
            s = np.full_like(folded_ima, fill_value=qi, dtype=np.float64)

        folded_ima = unit_scale(folded_ima)
        #print("Folded IMA",folded_ima, bar_duration)
        
            
        return self.measure(folded_ima,s,qi=qi,a=a)



class ABCRhythmTool():
    def __init__(self, unit_length = 1/8, min_note_val=1/64, meter="4/4"):
        
        # Note Extension Pattern (Reusable for notes, chords, and rests)
        note_extension_pattern = r"(?:\d+/\d+|/\d+|\d+|/|>{1,3}|<{1,3}|-)*"
        
        # Chord Pattern (Uses Note Extension)
       # chord_note_pattern = rf"(?<!:)\[[^\]]*\]{note_extension_pattern}"

        # Single Note Pattern (Uses Note Extension)
        #single_note_pattern = rf"[A-Ga-g][,']*{note_extension_pattern}"

        # Rest Pattern (Supports z and x, Uses Note Extension)
        rest_pattern = rf"[zx]{note_extension_pattern}"
        
        tuple_pattern = r"\(\d+(?::\d+)?(?::\d+)?"
        
        self.alpha_pattern = r"A-Ga-g"
        self.duration_pattern = r"(\d+/\d+|\d+/|\d+|/\d+|/)"
        self.decoration_pattern = r"!.*?!"

        # Apply decoration only before chords and notes
        decorated_prefix = rf"(?:\.|{self.decoration_pattern})*"

        # Chord: [CEG]2, [CE]3/2, etc.
        dec_chord_note_pattern = rf"{decorated_prefix}(?<!:)\[[^\]]*\]{note_extension_pattern}"

        # Single Note: A, B,,2, ^C'/2, etc.
        dec_single_note_pattern = rf"{decorated_prefix}[A-Ga-g][,']*{note_extension_pattern}"
        
        self.full_pattern = rf"({tuple_pattern})|({dec_chord_note_pattern})|({dec_single_note_pattern})|({rest_pattern})"
        

        # Regular expression pattern for headers, comments, decorations and quotes
        self.header_pattern = r"\[[A-Za-z]:[^\]]*\]"
        self.comment_pattern = r"(?m)(?<!\\)%.*$"
        self.decoration_pattern = r"!.*?!"
        self.quoted_pattern = r"\".*?\""

        self.unit_length = unit_length
        self.min_note_val = min_note_val
        self.meter = meter 
        self.sc = SyncopationScore(proto_rhythms=proto_rhythms)

        self.sc_a = 0.34
        

    def extract_broken_rhythms(self, abc_string, base_duration):
        """Extract uninterrupted groups of > and < from an ABC notation string, preserving order."""
        greater_groups = [(match.group(), match.start()) for match in re.finditer(r">+", abc_string)]
        less_groups = [(match.group(), match.start()) for match in re.finditer(r"<+", abc_string)]

        # Combine both lists of groups
        all_groups = greater_groups + less_groups

        # Sort by the starting index to preserve order
        all_groups.sort(key=lambda x: x[1])

        # Extract the groups in sorted order
        broken_rhythms = [group for group, _ in all_groups]
        big_duration = base_duration
        small_duration = base_duration
        ret = (big_duration,small_duration)
        if(broken_rhythms!=[]):
            #Discard all broken rhythm indicators before the last one
            last_rhythm = broken_rhythms[-1]
            #for each symbol in the element 
            factor = base_duration 
            n = len(last_rhythm)
            while(n>0):
                factor=factor/2
                big_duration = big_duration+factor
                small_duration = small_duration/2
                n-=1
            if("<" in last_rhythm): #switch orders
                ret = (small_duration,big_duration)
            else:
                ret = (big_duration, small_duration)
        return ret

    def parse_duration(self, duration_element):
        try:
            if("/" in duration_element):
                nom,denom = duration_element.split("/")[:2]
                if(nom==""):
                    nom=1
                if(denom==""):
                    denom=2
                denom = int(denom)
                nom = int(nom)
                return nom/denom
            else:
                return int(duration_element)
        except Exception as e:
            print(f"Error occured when parsing {duration_element} - {e}")
            return 1
   
    
    def create_duration(self,target_duration,unit_length=None,min_note_val=None):
        unit_length = unit_length if unit_length else self.unit_length
        min_note_val = min_note_val if min_note_val else self.min_note_val
        r_duration = target_duration*min_note_val
        ratio = r_duration/unit_length
        #print("Ratio",ratio)
        if(ratio==1):
            return ""
        else:
            frac = fractions.Fraction(ratio)
            nom = frac.numerator
            denom = frac.denominator
            if(nom==1):
                return f"/{denom}"
            if(denom==1):
                return f"{nom}"
            return f"{nom}/{denom}"
        


    def get_duration(self, element, previous_element=None, current_tuple=None,unit_length=None, min_note_val=None):
        #Return durations as multiples of the unit_length
        unit_length = unit_length if unit_length else self.unit_length
        min_note_val = min_note_val if min_note_val else self.min_note_val
        durations = re.findall(self.duration_pattern,element)
        #print(durations)
        if(len(durations)>1):
            raise(ValueError(f"Formatting error, several durations found in {element}"))
        elif(len(durations)==1):
            factor = self.parse_duration(durations[0])
            #print(factor,durations[0],self.unit_length,factor*self.unit_length)
            duration = factor*unit_length
        else:
            duration = unit_length #If no duration is specified this is the default duration
        
        # Adjust for tuplets (scale by q/p)
        if current_tuple:
            duration *= current_tuple[1] / current_tuple[0]
        
        if(previous_element!=None):
            previous_duration, duration = self.extract_broken_rhythms(previous_element, duration)
        
        
        duration, next_duration = self.extract_broken_rhythms(element, duration)
        
        duration=duration/min_note_val
        return duration
        


    def resolve_ties(self,current_symbol,prior_symbol):
        if(prior_symbol==None):
            return False
        if("-" in prior_symbol):
            csymbol_set = set(re.findall(self.alpha_pattern,current_symbol))
            psymbol_set = set(re.findall(self.alpha_pattern,prior_symbol))
            if(csymbol_set==psymbol_set):
                return True
        else:
            return False

    def clean_bar(self, abc, exclude_grace_notes=False):
        ''' remove parts of the ABC which is not notes or bar symbols by replacing them by spaces (in order to preserve offsets) '''

        repl_by_spaces = lambda m: ' ' * len(m.group(0))
        # replace non-note fragments of the text by replacing them by spaces (thereby preserving offsets), but keep also bar and repeat symbols
        abc = abc.replace('\r', '\n')
        abc = re.sub(r'(?s)%%beginps.+?%%endps', repl_by_spaces, abc)  # remove embedded postscript
        abc = re.sub(r'(?s)%%begintext.+?%%endtext', repl_by_spaces, abc)  # remove text
        abc = re.sub(self.comment_pattern, repl_by_spaces, abc) # remove comments
        abc = re.sub(r'\[\w:.*?\]', repl_by_spaces, abc)   # remove embedded fields
        abc = re.sub(r'(?m)^\w:.*?$', repl_by_spaces, abc) # remove normal fields
        abc = re.sub(r'\\"', repl_by_spaces, abc)          # remove escaped quote characters
        abc = re.sub(r'".*?"', repl_by_spaces, abc)        # remove strings
        abc = re.sub(r'!.+?!', repl_by_spaces, abc)        # remove ornaments like eg. !pralltriller!
        abc = re.sub(r'\+.+?\+', repl_by_spaces, abc)      # remove ornaments like eg. +pralltriller+
       # abc = re.sub(r'\[M:([\d]+\/[\d]+)\]', repl_by_spaces, abc)
        if exclude_grace_notes:
            abc = re.sub(r'\{.*?\}', repl_by_spaces, abc)  # remove grace notes
        return abc
    
    def clean_abc_file(self, abc_lines, abc_path):
        
        # delete blank lines
        abc_lines = [line for line in abc_lines if line.strip() != '']

        # unidecode
        abc_lines = unidecode_abc_lines(abc_lines)

        # clean information field
        abc_lines = remove_information_field(abc_lines=abc_lines, info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])

        # delete bar number annotations
        abc_lines = remove_bar_no_annotations(abc_lines)

        # delete \"
        for i, line in enumerate(abc_lines):
            if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
                continue
            else:
                if r'\"' in line:
                    abc_lines[i] = abc_lines[i].replace(r'\"', '')

        # delete text annotations with quotes
        for i, line in enumerate(abc_lines):
            quote_contents = re.findall(Quote_re, line)
            for quote_content in quote_contents:
                for barline in Barlines:
                    if barline in quote_content:
                        line = line.replace(quote_content, '')
                        abc_lines[i] = line

        # check bar alignment
        try:
            _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
            if not bar_no_equal_flag:
                print(abc_path, 'Unequal bar number')
                raise Exception
        except:
            raise Exception

        # deal with text annotations: remove too long text annotations; remove consecutive non-alphabet/number characters
        for i, line in enumerate(abc_lines):
            quote_matches = re.findall(r'"[^"]*"', line)
            for match in quote_matches:
                if match == '""':
                    line = line.replace(match, '')
                if match[1] in ['^', '_']:
                    sub_string = match
                    pattern = r'([^a-zA-Z0-9])\1+'
                    sub_string = re.sub(pattern, r'\1', sub_string)
                    if len(sub_string) <= 40:
                        line = line.replace(match, sub_string)
                    else:
                        line = line.replace(match, '')
            abc_lines[i] = line
        

        #count notes per bar across voices
        for i, line in enumerate(abc_lines):
            #print(i,line.split())
            quote_matches = re.findall(r'"[^"]*"', line)
            for match in quote_matches:
                if match == '""':
                    line = line.replace(match, '')
                if match[1] in ['^', '_']:
                    sub_string = match
                    pattern = r'([^a-zA-Z0-9])\1+'
                    sub_string = re.sub(pattern, r'\1', sub_string)
                    if len(sub_string) <= 40:
                        line = line.replace(match, sub_string)
                    else:
                        line = line.replace(match, '')
            abc_lines[i] = line
        return abc_lines
    
    
    def get_default_q(self,p, time_signature="4/4"):
        """Determines the default q value based on p and time signature."""
        compound_meters = {"6/8", "9/8", "12/8"}
        n = 3 if time_signature in compound_meters else 2
        if p in {2, 4, 8}: return 3
        if p in {3, 6}: return 2
        if p in {5, 7, 9}: return n
        return n  # Fallback case

    def parse_tuple_notation(self, tuple_str, time_signature="4/4"):
        """Expands (p), (p:q), or (p::r) into full (p:q:r) notation."""
        parts = tuple_str.strip("()").split(":")
        p = int(parts[0])
        q = int(parts[1]) if len(parts) > 1 and parts[1] else self.get_default_q(p, time_signature)
        r = int(parts[2]) if len(parts) > 2 and parts[2] else p
        return (p,q,r)

    import re

    def remove_grace_notes(self, abc_text):
        """
        Removes grace notes and ornaments from ABC notation.
        Specifically:
        - Removes any ~X pattern (where X is a single character).
        - Removes anything inside {} (grace note groups), including the braces.
        """
        # Remove ~X (tilde followed by a character, e.g. ~G or ~A/)
        abc_text = re.sub(r'~.', '', abc_text)

        # Remove {grace notes} including the braces (could be multiple notes or decorations)
        abc_text = re.sub(r'\{[^}]*\}', '', abc_text)

        return abc_text

    def extract_optional_single_meter(self,abc_text):
        """
        Extracts a single inline meter [M:d/d] from the ABC string.
        - Returns the meter as a string (e.g. '6/8') if exactly one is found.
        - Returns None if no inline meter is found.
        - Raises ValueError if more than one inline meter is found.
        """
        meters = re.findall(r'\[M:([\d]+\/[\d]+)\]', abc_text)
        if len(meters) > 1:
            raise ValueError(f"Multiple inline meters found: {meters}")
        return meters[0] if meters else None

    def extract_onsets_bar(self,bar_string, previous_element=None, position_offset = 0, unit_length=None,min_note_val=None,meter=None):        
        """
        Parses a single bar of ABC notation and returns its extracted elements and onsets
        """

        bar_meter = self.extract_optional_single_meter(bar_string)

        bar_string=self.clean_bar(bar_string)
        bar_string = self.remove_grace_notes(bar_string)

        unit_length=unit_length if unit_length else self.unit_length
        min_note_val = min_note_val if min_note_val else self.min_note_val
        self.meter = bar_meter if bar_meter else self.meter
        meter = meter if meter else self.meter
        
        elements = []
        current_tuple = None #keep track of current tuple
        tuple_counter = -1 #
        duration = None
        
        current_position = position_offset
        onsets = set()

       # print(unit_length,meter,min_note_val)
        for match in re.finditer(self.full_pattern, bar_string):
            #print(onsets,match, current_position)
            ntuple, chord, note, rest = match.groups()
            if ntuple:
                ntuple = self.parse_tuple_notation(ntuple,time_signature=meter)
                current_tuple = ntuple
                tuple_counter = current_tuple[2]
            else:
                current_element = chord or note or rest
                #print(current_element)
                duration = self.get_duration(current_element,
                                             previous_element=previous_element,
                                             current_tuple=current_tuple,
                                             unit_length=unit_length,
                                             min_note_val=min_note_val
                                             )
                tuple_counter-=1
                is_tie = self.resolve_ties(current_element,previous_element)
                if chord:
                    # Extract notes inside the chord and store them to avoid repeating
                    notes_in_chord = re.findall(r"[A-Ga-g][,']*", chord)  # Capture just the notes
                    if(not is_tie):
                        onsets.add(current_position)
                elif note:
                    #print("Add single note")
                    if(not is_tie):
                        onsets.add(current_position)                    
                elif rest:
                    pass
                previous_element = current_element
                current_position+=duration
            current_element = chord or note or rest or ntuple
            #print(onsets,current_element)
            if(tuple_counter<=0):
                current_tuple=None
                
            elements.append((current_element,duration))
            
        #print(elements)
        return onsets, previous_element,current_position
    

    def correct_single_bar(self,bar_string, previous_element=None,unit_length=None,min_note_val=None,meter=None):        
        """
        Parses a single bar of ABC notation and corrects the bar alignment
        """

        bar_meter = self.extract_optional_single_meter(bar_string)
        bar_string=self.clean_bar(bar_string)
        bar_string = self.remove_grace_notes(bar_string)

        unit_length=unit_length if unit_length else self.unit_length
        min_note_val = min_note_val if min_note_val else self.min_note_val
        self.meter = meter if meter else bar_meter if bar_meter else self.meter
        #self.meter = bar_meter if bar_meter else self.meter
        meter = meter if meter else self.meter
        #print(bar_string)
        elements = []
        current_tuple = None #keep track of current tuple
        tuple_counter = -1 #
        duration = None
        
        beats, beat_unit = map(int, self.meter.split("/"))  # e.g., 4/4 -> beats=4, beat_unit=4
        # Total duration of the bar in units of 1/max_denom notes
        bar_duration = int((beats / beat_unit) * 1/min_note_val)
        #print(self.meter,bar_duration)
        real_duration = 0
        tuple_id = 0
        broken_id = 0
        is_prev_broken = False

       # print(unit_length,meter,min_note_val)
        for match in re.finditer(self.full_pattern, bar_string):
            ntuple, chord, note, rest = match.groups()
            if ntuple:
                ntuple_arr = self.parse_tuple_notation(ntuple,time_signature=meter)
                current_tuple = ntuple_arr
                tuple_id+=1
                tuple_counter = current_tuple[2]
                elements.append((ntuple,0,current_tuple, tuple_id, False, 0))
            else:
                current_element = chord or note or rest
                #print(current_element)
                duration = self.get_duration(current_element,
                                             previous_element=previous_element,
                                             current_tuple=current_tuple,
                                             unit_length=unit_length,
                                             min_note_val=min_note_val
                                             )
                tuple_counter-=1
                previous_element = current_element
                real_duration+=duration 
                if(tuple_counter<=0):
                    current_tuple=None

                
                greater_groups = [(match.group(), match.start()) for match in re.finditer(r">+", current_element)]
                less_groups = [(match.group(), match.start()) for match in re.finditer(r"<+", current_element)]
                all_groups = greater_groups + less_groups
                broken_rhythms = [group for group, _ in all_groups]
                current_broken_id = broken_id
                #print(current_element, is_prev_broken)
                
                if(broken_rhythms!=[] or is_prev_broken):
                    is_broken = True
                    current_broken_id = broken_id
                    if(is_prev_broken): #Then this is the matching pair
                        broken_id+=1
                 #       print("Broken ids", broken_id,current_broken_id,current_element)
                        is_prev_broken=False
                    else:
                        is_prev_broken=True
                else:
                    is_broken=False
                    is_prev_broken = False  
                
                elements.append((current_element,duration,current_tuple, tuple_id, is_broken, current_broken_id))
        #print(elements)
        if(bar_duration !=real_duration):
            if(real_duration>bar_duration):
                #remove prior onsets untill target duration is reached (or overshot), then pad bar if necessary
                remaining_duration = real_duration-bar_duration
                while remaining_duration>0:
                    current_element, current_duration, current_tuple, current_tuple_id, is_broken, current_broken_id = elements.pop()
                    remaining_duration-=current_duration
                if(remaining_duration<0):
                    if(current_tuple): #If last removed element was in a tuple, remove entire tuple
                        _,removed_duration, next_tuple, next_tuple_id, next_is_broken, next_broken_id = elements[-1]
                        while next_tuple_id==current_tuple_id:
                            remaining_duration-=removed_duration
                            del elements[-1]
                            _,removed_duration, next_tuple, next_tuple_id, next_is_broken, next_broken_id = elements[-1]
                    elif(is_broken): #If last removed element was in a broken rhythm pair, remove broken partner.
                        _,removed_duration, next_tuple, next_tuple_id, next_is_broken, next_broken_id = elements[-1]
                        if(next_broken_id==current_broken_id and next_is_broken):
                            del elements[-1]
                            remaining_duration-=removed_duration
                    remaining_duration*=-1
            else:
                remaining_duration=bar_duration-real_duration
            #print(elements)

            remaining_duration =round(remaining_duration,2)
            if(remaining_duration!=0):
                #print("Duration remaining",remaining_duration)
                duration_appendix = self.create_duration(remaining_duration, unit_length,min_note_val)
                elements.append((f"x{duration_appendix}",remaining_duration,None,0,False,0))
                #print(elements,bar_duration,real_duration, remaining_duration)
                
        
            corrected_bar = ""
            for element,duration,_,_,_,_ in elements:
                corrected_bar = corrected_bar+element
            previous_element,_,_,_,_,_ = elements[-1]
            return corrected_bar, previous_element, False
        else:
            previous_element,_,_,_,_,_ = elements[-1]
            return bar_string, previous_element, True

    
    def units_in_bar(self) -> int:
        meter = self.meter           # e.g. "6/8"
        min_len = self.min_note_val    # e.g. 1/16 as a float
        beats, beat_note = map(int, meter.split("/"))
        meter_value = beats / beat_note
        return int(meter_value / min_len)

    def calculate_avg_onset_distance(self, onsets_per_voice_per_bar, meter_per_bar):
        onsets_per_bar = [set()]*len(onsets_per_voice_per_bar["V:1"])
        onset_distances = []
        for voice, onsetls in onsets_per_voice_per_bar.items():
            for i in range(0,len(onsetls)):
                cur_bar_onsets = onsetls[i]
                onsets_per_bar[i] = onsets_per_bar[i].union(cur_bar_onsets)


        
        assert len(onsets_per_bar)==len(meter_per_bar),"Unequal lengths of meter per bar and onsets per bar"
        for bar in range(0,len(onsets_per_bar)):
            current_meter = meter_per_bar[bar]
            beats, beat_unit = map(int, current_meter.split("/"))  # e.g., 4/4 -> beats=4, beat_unit=4
            # Total duration of the bar in units of 1/max_denom notes
            bar_duration = int((beats / beat_unit) * 1/self.min_note_val)
            n_onsets = 0

            cum_distance = 0
            if(len(onsets_per_bar)==1):
                cum_distance=bar_duration
            if(len(onsets_per_bar[bar])>1):
                cum_distance = 0
                onset_ls = list(onsets_per_bar[bar])
                onset_ls.sort()
                for i in range(1,len(onset_ls)):
                    prev_onset = onset_ls[i-1]
                    current_onset = onset_ls[i]
                    cum_distance += (current_onset - prev_onset)

                cum_distance = cum_distance/(len(onsets_per_bar[bar])-1)
            onset_distances.append(cum_distance)
        return onset_distances

    def calculate_syncopation_score(self,onsets_per_bar,offsets, meters_per_bar):
        syncopation_scores = []
        for i, onsets_bar in onsets_per_bar.items():
            onsets_bar = list(onsets_bar)
            subtracted_ls = list(set([max(0,x - offsets[i]) for x in onsets_bar]))
            subtracted_ls.sort()
            if(len(onsets_bar)>0):
                #print(i,meters_per_bar[i],subtracted_ls)
                score = self.sc.calculate_syncopation_score_bar(subtracted_ls, meters_per_bar[i], spectral=False, a=1, qi=None, resolution=1/self.min_note_val, bar_type="onsets")
                syncopation_scores.append(score)
            else:
                syncopation_scores.append(0)
        return syncopation_scores
    
    def arrange_ima(self,ima_s,meter_per_bar):
        current_offset = 0
        ima_arr = {}
        for bar,meter in meter_per_bar.items():
            beats, beat_unit = map(int, meter.split("/"))  # e.g., 4/4 -> beats=4, beat_unit=4

            # Total duration of the bar in units of 1/max_denom notes
            bar_duration = int((beats / beat_unit) * 1/self.min_note_val)
            ima_arr.update({bar:ima_s[current_offset:current_offset+bar_duration]})
            #print(bar, len(ima_s[current_offset:current_offset+bar_duration]),ima_s[current_offset:current_offset+bar_duration])
            current_offset+=bar_duration
            
        return ima_arr


    def get_treble_onsets(self,onsets_per_voice_per_bar,treble_voices):
        if(len(treble_voices)==0):
            return []
        treble_onsets = []
        for voice, onsetls in onsets_per_voice_per_bar.items():
            if(voice in treble_voices):
                for i in range(0,len(onsetls)):
                    treble_onsets+= [int(onset) for onset in onsetls[i]]
        treble_onsets.sort()
        return treble_onsets        


    
    def extract_unique_onsets(self,abc_path, rotated=False, voice_target=None):

        with open(abc_path, 'r', encoding='utf-8') as f:
                abc_lines = f.readlines()
        if(rotated):
            metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = extract_barline_and_bartext_dict_rotated(abc_lines)
        
        else:
            abc_lines = self.clean_abc_file(abc_lines,abc_path)
            interleaved_abc = rotate_abc(abc_lines)
            #print(interleaved_abc, abc_lines)

            metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = extract_barline_and_bartext_dict_rotated(interleaved_abc)
        voices = list(prefix_dict.keys())
        
        #print(metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict)
        treble_voices = [voice_target]
        for entry in metadata_lines:
            if(entry.startswith("L:")):
                    rmatch = re.search(r"1/(?:2|4|8|16|32|64)",entry)
                    if(not rmatch):
                        self.unit_length=1/8
                    else:
                        a,b=rmatch.group().split("/")
                        self.unit_length=int(a)/int(b)
            if(entry.startswith("M:")):
                rmatch = re.search(r"\d+/\d+",entry)
                if(not rmatch):
                    self.meter = "4/4"
                else:
                    self.meter = rmatch.group()
            """ alternative for voice targeting (i.e get all treble clef voices)
            if(entry.startswith("V:")):
                # Define the pattern
                #entry=entry.strip()
                treble_pattern= r"^\s*V:(\d+)\s+treble\s*$"
                match = re.match(treble_pattern, entry)
                if match:
                    digit = match.group(1)
                    treble_voices.append(f"V:{digit}")
                #else:
                    #print("No match found in ",entry)
            """
        #print(f"Extracting onsets from {abc_path}, with meter {self.meter} and unit_length {self.unit_length}")

        global_onsets = set()
        onset_count_per_voice_per_bar = {}
        onsets_per_voice_per_bar ={}
        onsets_per_bar = {}
        treble_onsets_per_bar = {}
        meter_per_bar = {}
        voices=[voices[0]]

        for voice in voices:
            current_position = 0
            current_voice_onset_count= []
            current_voice_onsets = []

            #print(voices)
            previous_element=None
            bar_count = 0
            baroffsets = []
            for i, bar in enumerate(bar_text_dict[voice]):
                if(i in meter_per_bar):
                    self.meter = meter_per_bar[i]
                
                tmp_offset = current_position
                onsets, previous_element,current_position = self.extract_onsets_bar(bar,
                                                                                    previous_element= previous_element,
                                                                                    position_offset=current_position,
                                                                                    )
                current_voice_onset_count.append(len(onsets))
                current_voice_onsets.append(onsets)
                current_bar_onsets = onsets_per_bar.get(i,set())
                baroffsets.append(tmp_offset)
                current_bar_onsets = current_bar_onsets.union(onsets)
                onsets_per_bar.update({i:current_bar_onsets})
                
                if(voice in treble_voices):
                    treble_current_bar_onsets = treble_onsets_per_bar.get(i,set())
                    treble_current_bar_onsets = treble_current_bar_onsets.union(onsets)
                    treble_onsets_per_bar.update({i:treble_current_bar_onsets})   

                meter_per_bar.update(({i:self.meter}))
                global_onsets = global_onsets.union(onsets)
            onset_count_per_voice_per_bar.update({voice:current_voice_onset_count})
            onsets_per_voice_per_bar.update({voice:current_voice_onsets})

        ls = list(global_onsets)
        ls.sort()

        units_per_bar = self.units_in_bar()
        densities = self.calculate_avg_onset_distance(onsets_per_voice_per_bar,meter_per_bar)
        onsets_int = [int(onset) for onset in ls]

        #print(onsets_int)
        treble_onsets=self.get_treble_onsets(onsets_per_voice_per_bar,treble_voices)
        
        ima_s =  pyinmean.get_normalized_ima(onsets_int,True)
        ima_m =  pyinmean.get_normalized_ima(onsets_int,False)

        if(len(treble_onsets)>3):
            #print(treble_onsets)
            ima_s_treble = pyinmean.get_normalized_ima(treble_onsets,True)
            ima_m_treble = pyinmean.get_normalized_ima(treble_onsets,False)
            ima_m_treble = self.sc.extend_metric_weight(ima_m_treble,treble_onsets)

        else:
            ima_s_treble = []
            ima_m_treble = []
        
        pr = sum(ima_m)
        ima_m = self.sc.extend_metric_weight(ima_m,onsets_int)
        pos = sum(ima_m)
        assert pr==pos, f"Unequal inner metric weight sums pre {pr} and post {pos} extension"

        #assert(len(ima_m)==len(ima_s)),f"Unqeual Lengths Spectral: {len(ima_s)} vs Metric:{len(ima_m)}"
        ima_s_arr = self.arrange_ima(ima_s,meter_per_bar)
        ima_m_arr = self.arrange_ima(ima_m,meter_per_bar)

        ima_s_arr_treble = self.arrange_ima(ima_s_treble,meter_per_bar)
        ima_m_arr_treble = self.arrange_ima(ima_m_treble,meter_per_bar)


        sync = self.calculate_syncopation_score(onsets_per_bar,baroffsets,meter_per_bar)
        
        #sync=[]
        return {"onsets":ls,
                "densities":densities,
                "spectral_arranged":ima_s_arr, 
                "metric_arranged":ima_m_arr, 
                "spectral_arranged_treble":ima_s_arr_treble, 
                "metric_arranged_treble":ima_m_arr_treble, 
                "meter":self.meter,
                "syncopation":sync
                }

    def correct_bar_alignment(self, abc_lines, rotated):
        if(rotated):
            metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = extract_barline_and_bartext_dict_rotated(abc_lines)
        
        else:
            abc_lines = self.clean_abc_file(abc_lines,abc_path)
            interleaved_abc = rotate_abc(abc_lines)
            #print(interleaved_abc, abc_lines)

            metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = extract_barline_and_bartext_dict_rotated(interleaved_abc)
        
        voices = bar_text_dict.keys()
        meter_per_bar = {}

        new_bar_text_dict = {}

        error_count_dict = {}
        for voice in voices:
            n_bars = []
            prev_element=None
            for i, bar in enumerate(bar_text_dict[voice]):
                if(i in meter_per_bar):
                    self.meter = meter_per_bar[i]
                
                corr_bar ,prev_element, no_error = self.correct_single_bar(bar,prev_element)
                #print(corr_bar)
                n_bars.append(corr_bar)
                meter_per_bar.update(({i:self.meter}))

                if(not no_error):
                    error_count_dict.update({i:1})
            new_bar_text_dict.update({voice:n_bars})
        
        abc_lines = merge_barline_and_bartext_dict(metadata_lines,prefix_dict,left_barline_dict,new_bar_text_dict,right_barline_dict)

        interleaved_abc = rotate_abc(abc_lines)
        return interleaved_abc, len(error_count_dict), len(meter_per_bar)

def test_bar_onset_calculation(tools):
    """
    Testing bar-wise duration and onset calculations
    """
    test_cases = [
        ("",1/8,1/64,"4/4",set(),"Empty Case"),
        ("DDDD",1/8,1/8,"4/4",{0,1,2,3},"eight notes"),
        ("D1/4E/4 F/2 G B2 C4",1/8,1/32,"4/4",{0,1,2,4,8,16},"Simple fractions"),
        ("D>D D<D z2<D2",1/8,1/16,"4/4",{0,3,4,5,10},"Rests and broken rhythms"),
        ("(3zDD z2 D2>>D2",1/8,1/48,"4/4",{4,8,24,45},"Rests and Triplets"),
        ("(3D>DD (3d/2d/2d/2 D DD DD",1/8,1/48,"4/4",{0,6,8,12,14,16,18,24,30,36,42},"Broken and 16th note Triplets"),
        ("(2dd ddd",1/8,1/48,"6/8",{0,9,18,24,30},"Compound duplets"),
        ("F/A/F/E/D/E/ F/A/F/E/D/E/",1/8,1/16,"6/8",{0,1,2,3,4,5,6,7,8,9,10,11},"Meine Ruhe"),
        ("[CEG]2 [FA]2 [DGB]2>D2", 1/8, 1/32, "4/4", {0, 8, 16, 28}, "Chords with different durations and broken rhythms"),
        ("D-D2 E E-E [FA]-A", 1/8, 1/8, "4/4", {0, 3,4, 6}, "Tied notes and broken rhythm"),
        ("(3[CE]F[GA] BG-  G4", 1/8, 1/48, "4/4", {0, 4, 8, 12, 18}, "Chords inside triplets"),
        ("(5:4:5 D-DDDD- D2 D2", 1/8, 1/60, "4/4", {0,12,18,24,45}, "Tuplet with ties"),
    ]

    for element, unit_length, min_note_val, meter, expected_onsets, name in test_cases:
        onsets,_,_ = tools.extract_onsets_bar(element,previous_element=None,unit_length=unit_length,min_note_val=min_note_val,meter=meter,position_offset=0)
        assert onsets==expected_onsets,f"Unexpected onsets {onsets} vs {expected_onsets} in {name}"
    print("Succesfully bar-wise onset calculation!")


def test_duration_calculation(tools):
    """
    Testing duration calculations inclusing complex fractions, broken rhythms and tuples
    """
    test_cases = [
        ("D/",None,None,1/8,1/16,1,"simple fraction"),
        ("D",None,None,1/8,1/8,1,"standart duration"), 
        ("D2",None,None,1/8,1/8,2,"multiple duration"),
        ("D2",None,None,1/8,1/16,4,"multiple duration/different base"), 
        ("D1/2",None,None,1/8,1/16,1,"fractional duration"), 
        ("D/2",None,None,1/8,1/16,1,"fractional duration short notatino"),
        ("D>",None,None,1/8,1/16,3,"dotted eigth note"), 
        ("D2>",None,None,1/8,1/16,6,"dotted quarter note"), 
        ("D/2>",None,None,1/8,1/32,3,"dotted sixteenth note"),
        ("D","D>",None,1/8,1/16,1,"dotted eight note preceeding"), 
        ("D2","D2>",None,1/8,1/16,2,"dotted quarter note preceeding"),
        ("D4","D4>",None,1/8,1/16,4,"dotted half note preceeding"),
        ("D4","D4",None,1/8,1/16,8,"standard half note"),
        ("D4","D4>>",None,1/8,1/16,2,"double dotted half note preceeding"),
        ("D4>>","D4",None,1/8,1/16,14,"double dotted half note"),
        ("D4<<","D4",None,1/8,1/16,2,"reverse double dotted half note"),
        ("D4>","D4>",None,1/8,1/16,6,"elongation row (very unlikely)"),
        ("D2","D2",None,1/8,1/24,6,"standard quarter notes different time subdivision"),
        ("D2","D2",(3,2,3),1/8,1/24,4,"standard quarter-note triplets"),
        ("D","D",(3,2,3),1/8,1/24,2,"standard eight note triplets"),
        ("D4","D",(3,2,2),1/8,1/24,8,"mixed value triplets"),
        ("D","D",(2,3,2),1/8,1/16,3,"compound meter duplets"),
    ]

    for element, previous_element, n_tuple, unit_length, min_note_val, expected_duration, name in test_cases:
        duration = tools.get_duration(element,previous_element=previous_element,current_tuple=n_tuple,unit_length=unit_length,min_note_val=min_note_val)
        assert duration==expected_duration,f"Unexpected duration {duration} vs {expected_duration} in {name}"
    print("Succesfully tested duration calculation!")


def test_parse_tuple_notation(tools):
    """Test cases for tuple notation parsing."""
    test_cases = [
        ("(3", "4/4", (3, 2, 3)),  # Default q=2, r=p
        ("(3::", "6/8", (3, 2, 3)),  # Default q=3 in compound meter
        ("(5:3", "4/4", (5, 3, 5)),  # Given q, default r=p
        ("(7::4", "6/8", (7, 3, 4)),  # Default q=3, r=4
        ("(9", "4/4", (9, 2, 9)),  # Default q=2 for n
        ("(9", "9/8", (9, 3, 9)),  # Default q=3 for compound meter
    ]
    
    for tup_str, time_sig, expected in test_cases:
        result = tools.parse_tuple_notation(tup_str, time_sig)
        assert result == expected, f"Failed on {tup_str} with {time_sig}: got {result}, expected {expected}"
    
    print("Succesfully tested tuple parsing!")



def test_header_parser():
    strings = [
        ("L:1/8",1/8),
        ("L:1/16",1/16),
        ("L:1/4",1/4),
        ("L:1/5",1/8),
        ("L:5/8",1/8),
        ("L:5fg/dsas",1/8),
    ]
    for entry,result in strings:
        rmatch = re.search(r"1/(?:2|4|8|16|32|64|128|256|512|1024)",entry)
        if(not rmatch):
            f=1/8
        else:
            a,b=rmatch.group().split("/")
            f=int(a)/int(b)
        assert f==result, f"{result} is not given by {f} in {entry}"
    print("Succesfully tested header parsing!")


def test_bar_corrections(tools):
    """
    Testing bar-wise duration and onset calculations
    """
    test_cases = [
        ("DDD",1/4,1/8,"4/4","DDDx","quarter notes"),
        (".D .D .D",1/4,1/8,"4/4",".D.D.Dx","quarter notes"),

        ("DDDDDDD",1/8,1/8,"4/4","DDDDDDDx","eigth note"),
        ("DDDDDD",1/8,1/8,"4/4","DDDDDDx2","quarter note rest"),
        ("[M:3/4]DDDDD",1/8,1/8,None,"DDDDDx","eigth note in defined 3/4 time"),
        ("D2D2D2D2D2",1/8,1/8,"4/4","D2D2D2D2","shortening measure"),
        ("D2D2D2D4",1/8,1/8,"4/4","D2D2D2x2","shortening measure and padding"),
        ("D2 D2 D2 D2 (3DDD",1/8,1/8,"4/4","D2D2D2D2","shortening measure and padding"),
        ("D2 D2 D2 D (3DDD",1/8,1/8,"4/4","D2D2D2Dx","breaking up tuplets"),
        ("(3DDD D2 D2 D (3DDD",1/8,1/8,"4/4","(3DDDD2D2Dx","breaking up multi tuplets"),
        ("D2>D2 D2 D D>D",1/8,1/8,"4/4","D2>D2D2Dx","breaking up broken rhythms"),
        ("D2<D2 D2 D D<D",1/8,1/8,"4/4","D2<D2D2Dx","breaking up broken rhythms rev"),
    ]

    for element, unit_length, min_note_val, meter, expected_onsets, name in test_cases:
        onsets,_,_ = tools.correct_single_bar(element,previous_element=None,unit_length=unit_length,min_note_val=min_note_val,meter=meter)
        assert onsets==expected_onsets,f"Unexpected onsets {onsets} vs {expected_onsets} in {name}"
    print("Succesfully bar-wise corrections")




if __name__=="__main__":
    abc_path = "data/outputs/Lieder/L3_spect_inattn/sync-0-lc4945954_0.abc-15.242442846298218_2.abc"
    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()
    
    tools = ABCRhythmTool(min_note_val=1/48)
    onset_dic= tools.extract_unique_onsets(abc_path, rotated=True,voice_target="V:1")
    
    print("Syncopation", onset_dic["syncopation"])
    print("Density", onset_dic["densities"])

    for key,val in onset_dic["spectral_arranged_treble"].items():
        val = [np.round(v) for v in val]

        met = onset_dic["metric_arranged_treble"][key]

        met = [np.round(m,1) for m in met]


        print("Spect",val)
        print("Metri",met)
        print(len(val),len(met))
    
    test_bar_onset_calculation(tools)
    test_duration_calculation(tools)
    test_parse_tuple_notation(tools)
    test_header_parser()



