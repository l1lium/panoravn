import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import create_matcher
from core.io import load_config

def test_matcher_interchangeability():
    test_dir = "data/test"
    assert os.path.exists(test_dir), f"{test_dir} does not exist"

    image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(image, (100, 150), 40, (255, 100, 100), 3)
    cv2.circle(image, (100, 150), 35, (200, 50, 50), -1)
    cv2.circle(image, (300, 150), 50, (100, 255, 100), 3)
    cv2.circle(image, (300, 150), 45, (50, 200, 50), -1)
    cv2.rectangle(image, (50, 250), (150, 350), (200, 200, 0), 3)
    cv2.rectangle(image, (55, 255), (145, 345), (150, 150, 0), -1)
    cv2.rectangle(image, (250, 250), (350, 300), (0, 200, 200), 3)
    cv2.rectangle(image, (255, 255), (345, 295), (0, 150, 150), -1)

    image_left = image[:, :280].copy()
    image_right = image[:, 120:].copy()

    sift_config = load_config("configs/baseline.yaml")
    sift_matcher = create_matcher(sift_config["matcher"])

    points_a_sift, points_b_sift, matches_sift = sift_matcher.match(
        cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    )
    assert isinstance(points_a_sift, np.ndarray), "SIFT should return numpy array for points_a"
    assert isinstance(points_b_sift, np.ndarray), "SIFT should return numpy array for points_b"
    assert isinstance(matches_sift, list), "SIFT should return list for matches"
    assert points_a_sift.shape[1] == 2, "Points should have (x, y) format"
    assert len(matches_sift) > 0, "SIFT should find some matches in synthetic image"

    orb_config = load_config("configs/baseline.yaml")
    orb_config["matcher"]["name"] = "orb"
    orb_config["matcher"]["detector"] = "orb"
    orb_matcher = create_matcher(orb_config["matcher"])

    points_a_orb, points_b_orb, matches_orb = orb_matcher.match(
        cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    )
    assert isinstance(points_a_orb, np.ndarray), "ORB should return numpy array for points_a"
    assert isinstance(points_b_orb, np.ndarray), "ORB should return numpy array for points_b"
    assert isinstance(matches_orb, list), "ORB should return list for matches"

    assert points_a_sift.shape == (len(matches_sift), 2)
    assert points_b_sift.shape == (len(matches_sift), 2)
    assert points_a_orb.shape == (len(matches_orb), 2)
    assert points_b_orb.shape == (len(matches_orb), 2)
