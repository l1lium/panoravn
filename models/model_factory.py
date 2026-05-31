from models.classical_matcher import ClassicalMatcher
from models.loftr_matcher import LoFTRMatcher

def create_matcher(config: dict):
    name = config.get("name", "sift").lower()
    if name in {"sift", "orb"}:
        return ClassicalMatcher(config)
    if name == "loftr":
        return LoFTRMatcher(config)
    raise ValueError(f"Unsupported matcher name: {name}")
