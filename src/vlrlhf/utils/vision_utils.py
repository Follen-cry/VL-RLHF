# src/vlrlhf/utils/vision_utils.py
from typing import Any

def _maybe_call(x):
    return x() if callable(x) else x

def get_vision_tower(model: Any):
    """
    Return the vision tower/module from a wide range of VL model layouts.
    Works with Qwen-VL, LLaVA, CLIP-based models, etc.
    """
    # Many HF models put the guts under `.model`
    root = getattr(model, "model", model)

    # 1) Common direct attributes or accessors
    candidates = [
        "get_vision_tower",   # LLaVA style accessor
        "vision_tower",       # attr
        "visual",             # Qwen-VL often uses .visual
        "vision_model",       # CLIPVisionModel / similar
        "vision_encoder",     # some custom repos
        "image_encoder",      # generic
        "clip_vision_model",  # CLIP naming
    ]
    for name in candidates:
        if hasattr(root, name):
            return _maybe_call(getattr(root, name))

    # 2) Heuristic search over submodules for a vision backbone
    vision_class_names = {
        "VisionTower",
        "CLIPVisionModel",
        "CLIPVisionTransformer",
        "Qwen2_5_VisionTransformer",
        "Qwen2VisionTransformer",
        "SiglipVisionModel",
        "SiglipVisionTransformer",
        "EVA2VisionTransformer",
    }
    for _, mod in root.named_modules():
        if mod.__class__.__name__ in vision_class_names:
            return mod

    raise RuntimeError(
        "Could not locate a vision tower on this model. "
        "Checked common attributes and known submodules."
    )