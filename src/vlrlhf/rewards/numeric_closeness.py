# src/vlrlhf/rewards/numeric_closeness.py
import math, re, torch, os
from typing import Sequence, Optional, Dict, Any, Union

# Regex to grab first float
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")

# Default configs per task
TASK_CONFIGS = {
    "Aesthetics":      {"scale": 1.0, "metric": "l2", "penalty": -1.0, "range": (0.0, 10.0)},
    "Funniness":       {"scale": 1.0, "metric": "l2", "penalty": -1.0, "range": (0.0, 10.0)},
    "Memorability":    {"scale": 1.0, "metric": "l2", "penalty": -1.0, "range": (0.0, 1.0)},
    "Emotional_Valence": {"scale": 1.0, "metric": "l2", "penalty": -1.0, "range": (-3.0, 3.0)},
}

def _parse_first_number(text: Optional[str]) -> Optional[float]:
    if not isinstance(text, str): 
        return None
    m = _NUM.search(text)
    return float(m.group(0)) if m else None

def _compute_one(pred: float, tgt: float, metric: str) -> float:
    if metric == "l1":
        return -abs(pred - tgt)
    d = pred - tgt
    return -(d * d)

class NumericClosenessRewarder(torch.nn.Module):
    """
    Callable rewarder:
        rewards = rewarder(pred_texts, targets, context=batch)
    Expects:
      - pred_texts: list[str] decoded model outputs
      - targets: list[float] ground truth scores
      - context: dict with "image" key (path strings) to infer task
    """
    def __init__(self, metric="l2"):
        super().__init__()
        self.default_metric = metric

    @torch.no_grad()
    def __call__(
        self,
        pred_texts: Sequence[str],
        targets: Union[Sequence[float], torch.Tensor],
        context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().float().cpu().tolist()

        image_paths = context.get("image", []) if context else []
        preds = [_parse_first_number(t) for t in pred_texts]

        rewards = []
        for pred, tgt, img in zip(preds, targets, image_paths):
            # infer task from image path (dirname before first /)
            task = os.path.normpath(img).split(os.sep)[0] if img else "Aesthetics"
            cfg = TASK_CONFIGS.get(task, {"scale":1.0, "metric":self.default_metric, "penalty":-1.0, "range":(0,1)})

            a, b = cfg["range"]
            # clamp tgt to task range
            tgt = max(a, min(b, float(tgt)))
            if pred is None or math.isnan(pred):
                rewards.append(cfg["penalty"])
                continue
            pred = max(a, min(b, float(pred)))

            r = _compute_one(pred, tgt, cfg["metric"]) * cfg["scale"]
            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float32)
