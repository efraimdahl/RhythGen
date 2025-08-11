from safetensors.torch import load_file
import torch
from typing import Optional

def load_composite_state_dict(
    model: torch.nn.Module,
    main_path: str,
    control_encoder_path: Optional[str] = None,
    from_torch=False,
) -> dict:
    """
    Load composite state dict from main_path and optionally control_encoder_path.
    Assumes safetensors load_file returns flat dict of parameter tensors.
    """
    # Load main checkpoint flat dict
    print(f"Loading main checkpoint from {main_path}")
    if(not from_torch):
        main_ckpt = load_file(main_path)
    else:
        checkpoint = torch.load(main_path, map_location='cpu')
        print(f'Loaded Checkpoint from epoch {checkpoint["best_epoch"]} after a total of {checkpoint["epoch"]} epochs with evaluation loss {checkpoint["min_eval_loss"]}')
        main_ckpt = checkpoint["model"]
    print(main_ckpt.keys())

    # If checkpoint contains nested 'state_dict' key, unwrap it
    if "state_dict" in main_ckpt and isinstance(main_ckpt["state_dict"], dict):
        main_state = main_ckpt["state_dict"]
    else:
        main_state = main_ckpt  # already flat dict

    # Load control_encoder checkpoint flat dict
    control_state = {}
    if control_encoder_path:
        print(f"Loading control_encoder checkpoint from {control_encoder_path}")
        ctrl_ckpt = load_file(control_encoder_path)
        if "state_dict" in ctrl_ckpt and isinstance(ctrl_ckpt["state_dict"], dict):
            ctrl_raw = ctrl_ckpt["state_dict"]
        else:
            ctrl_raw = ctrl_ckpt

        # Remap keys from encoder.* → control_encoder.*
        control_state = {
            k.replace("encoder.", "control_encoder.", 1): v
            for k, v in ctrl_raw.items()
            if k.startswith("encoder.")
        }

    model_state = model.state_dict()
    composite_state = {}

    for key in model_state:
        if key in main_state:
            composite_state[key] = main_state[key]
            #print("loading ", key, "from main")
        elif key in control_state and model_state[key].shape == control_state[key].shape:
            composite_state[key] = control_state[key]
            #print("loading ", key, "from control state")
        else:
            #print("missing key", key)
            # leave missing to trigger load_state_dict missing key report
            pass

    missing_keys, unexpected_keys = model.load_state_dict(composite_state, strict=False)

    print(f"Total loaded parameters: {len(composite_state)}")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    return composite_state
