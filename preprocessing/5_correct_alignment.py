import os
import re
from collections import defaultdict
import tqdm
from utils import check_alignment_rotated
from abc_rhythm import ABCRhythmTool

# -----------------------
# Configuration
# -----------------------
tool = ABCRhythmTool(min_note_val=1/48)

FILE_PATH = "data/outputs/Lieder/L7_metric_xattn"
TARGET_PATH = "data/outputs/Lieder/L7_metric_xattn"

VALID_GROUPS = {"0", "1", "3"}
FLAGS = ["barline_equal_flag", "bar_no_equal_flag", "bar_dur_equal_flag"]

# Nested defaultdict factories
ResultsDict = lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"true": 0, "false": 0})))
PropsDict = lambda: defaultdict(lambda: defaultdict(lambda: {"missed": 0, "bars": 0}))


# -----------------------
# Helpers
# -----------------------
def extract_group_from_filename(filename: str) -> str | None:
    """Extract group ID from filename if valid."""
    try:
        parts = re.split(r"[-,_]", filename)
        candidate = parts[1]
        if candidate in VALID_GROUPS:
            return candidate
    except Exception as e:
        print(f"Warning: could not extract group from {filename}: {e}")
    return None


def process_single_file(file_path: str, target_dir: str, group: str | None,
                        results: dict, prop_results: dict, subdir_label: str) -> None:
    """
    Process a single .abc file:
    - Run alignment checks.
    - Correct misaligned bars if needed.
    - Save corrected file to target directory.
    - Update results and proportion tracking.
    """
    try:
        with open(file_path, "r") as file:
            abc_lines = file.readlines()

        barline_equal_flag, bar_no_equal_flag, bar_dur_equal_flag, num_bars = check_alignment_rotated(abc_lines)

        # Save or correct file
        if bar_dur_equal_flag:
            missed = 0
        else:
            abc_lines, missed, num_bars = tool.correct_bar_alignment(abc_lines, rotated=True)

        os.makedirs(target_dir, exist_ok=True)
        target_file_path = os.path.join(target_dir, os.path.basename(file_path))
        with open(target_file_path, "w") as outfile:
            outfile.writelines(abc_lines)

        # Update flag results
        flags = {
            "barline_equal_flag": barline_equal_flag,
            "bar_no_equal_flag": bar_no_equal_flag,
            "bar_dur_equal_flag": bar_dur_equal_flag,
        }

        update_targets = ["Total"] + ([group] if group else [])
        for g in update_targets:
            for key, val in flags.items():
                results[subdir_label][g][key]["true" if val else "false"] += 1
            prop_results[subdir_label][g]["missed"] += missed
            prop_results[subdir_label][g]["bars"] += num_bars

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def process_directory(abc_dir: str, subdir_label: str,
                      results: dict, prop_results: dict) -> None:
    """Process all .abc files in a given directory."""
    abc_files = [f for f in os.listdir(abc_dir) if f.endswith(".abc")]
    target_subdir = os.path.join(TARGET_PATH, subdir_label)

    for filename in tqdm.tqdm(abc_files, desc=f"Processing {subdir_label}", leave=False):
        file_path = os.path.join(abc_dir, filename)
        group = extract_group_from_filename(filename)
        process_single_file(file_path, target_subdir, group, results, prop_results, subdir_label)


def print_results(results: dict, prop_results: dict) -> None:
    """Print false proportions and missed bar proportions."""
    print("\nFalse Proportions by Directory and Group:")
    for subdir in sorted(results):
        print(f"\nDirectory: {subdir}")
        for group in ["0", "1", "3", "Total"]:
            if group not in results[subdir]:
                continue
            print(f"  Group {group}:")
            for key in FLAGS:
                true_count = results[subdir][group][key]["true"]
                false_count = results[subdir][group][key]["false"]
                total = true_count + false_count
                proportion = false_count / total if total > 0 else 0
                print(f"    {key}: {proportion:.3f} (False: {false_count}, True: {true_count})")

            total_missed = prop_results[subdir][group]["missed"]
            total_bars = prop_results[subdir][group]["bars"]
            proportion_missed = total_missed / total_bars if total_bars > 0 else 0
            print(f"    Missed bar proportion: {proportion_missed:.3f}")


# -----------------------
# Main execution
# -----------------------
def main():
    results = ResultsDict()
    prop_results = PropsDict()

    entries = os.listdir(FILE_PATH)
    has_subdirs = any(os.path.isdir(os.path.join(FILE_PATH, e)) for e in entries)

    if has_subdirs:
        for subdir in sorted(entries):
            subdir_path = os.path.join(FILE_PATH, subdir)
            if os.path.isdir(subdir_path):
                process_directory(subdir_path, subdir, results, prop_results)
    else:
        process_directory(FILE_PATH, "root", results, prop_results)

    print_results(results, prop_results)


if __name__ == "__main__":
    main()
