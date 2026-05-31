import numpy as np
import pytest
import cv2
from models.model_factory import create_matcher

@pytest.mark.skipif(
    True,
    reason="LoFTR requires kornia. Install separately or skip."
)
def test_loftr_matcher_instantiation():
    """Test LoFTR matcher can be created from config."""
    config = {"name": "loftr", "device": "cpu"}
    matcher = create_matcher(config)
    assert matcher is not None

@pytest.mark.skipif(
    True,
    reason="LoFTR requires kornia. Install separately or skip."
)
def test_loftr_matcher_interface():
    """Test LoFTR returns consistent interface."""
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(image, (100, 100), 30, 255, -1)

    config = {"name": "loftr", "device": "cpu"}
    matcher = create_matcher(config)
    points_a, points_b, matches = matcher.match(image, image)

    assert isinstance(points_a, np.ndarray)
    assert isinstance(points_b, np.ndarray)
    assert isinstance(matches, list)
    assert points_a.shape[1] == 2
    assert points_b.shape[1] == 2
    assert points_a.shape == points_b.shape

@pytest.mark.skipif(
    True,
    reason="LoFTR requires kornia. Install separately or skip."
)
def test_loftr_matcher_none_input():
    """Test LoFTR handles None input gracefully."""
    config = {"name": "loftr", "device": "cpu"}
    matcher = create_matcher(config)
    points_a, points_b, matches = matcher.match(None, None)

    assert points_a.shape == (0, 2)
    assert points_b.shape == (0, 2)
    assert matches == []

def test_loftr_config_loading():
    """Test LoFTR config can be loaded from file."""
    from core.io import load_config
    config = load_config("configs/loftr.yaml")
    assert config["matcher"]["name"] == "loftr"
    assert "confidence_threshold" in config["matcher"]

def test_matcher_factory_supports_loftr():
    """Test factory can create LoFTR matcher without errors."""
    config = {"name": "loftr"}
    try:
        matcher = create_matcher(config)
        assert matcher is not None
    except ImportError:
        pytest.skip("LoFTR dependencies not available")
