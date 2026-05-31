import cv2
import numpy as np
from models.model_factory import create_matcher

def test_classical_matcher_interface_orb():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 100), 30, (255, 255, 255), -1)
    matcher = create_matcher({"name": "orb", "detector": "orb", "ratio_test": 0.75, "cross_check": False})
    points_a, points_b, matches = matcher.match(image, image)
    assert points_a.shape == points_b.shape
    assert points_a.shape[0] == len(matches)
    assert points_a.shape[0] > 0

def test_matcher_returns_zero_on_empty_input():
    matcher = create_matcher({"name": "orb", "detector": "orb", "ratio_test": 0.75, "cross_check": False})
    points_a, points_b, matches = matcher.match(None, None)
    assert points_a.shape == (0, 2)
    assert points_b.shape == (0, 2)
    assert matches == []
