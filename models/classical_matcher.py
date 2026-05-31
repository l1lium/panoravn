import cv2
import numpy as np

from models.base_matcher import MatcherInterface

class ClassicalMatcher(MatcherInterface):
    def __init__(self, config: dict):
        self.config = config
        detector_name = config.get("detector", "sift").lower()
        self.detector = self._build_detector(detector_name)
        self.cross_check = bool(config.get("cross_check", False))
        self.ratio_test = float(config.get("ratio_test", 0.75))

    def _build_detector(self, detector_name: str):
        if detector_name == "sift":
            if hasattr(cv2, "SIFT_create"):
                return cv2.SIFT_create()
            if hasattr(cv2, "xfeatures2d"):
                return cv2.xfeatures2d.SIFT_create()
            raise RuntimeError("SIFT is not available in this OpenCV build")
        if detector_name == "orb":
            return cv2.ORB_create(nfeatures=2000)
        raise ValueError(f"Unsupported detector: {detector_name}")

    def match(self, image_a, image_b):
        if image_a is None or image_b is None:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), []

        keypoints_a, descriptors_a = self.detector.detectAndCompute(image_a, None)
        keypoints_b, descriptors_b = self.detector.detectAndCompute(image_b, None)
        if descriptors_a is None or descriptors_b is None or len(keypoints_a) == 0 or len(keypoints_b) == 0:
            return np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), []

        if self.cross_check:
            matcher = cv2.BFMatcher(cv2.NORM_L2 if descriptors_a.dtype == np.float32 else cv2.NORM_HAMMING, crossCheck=True)
            raw_matches = matcher.match(descriptors_a, descriptors_b)
            raw_matches = sorted(raw_matches, key=lambda x: x.distance)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2 if descriptors_a.dtype == np.float32 else cv2.NORM_HAMMING)
            knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
            raw_matches = []
            for m_n in knn_matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < self.ratio_test * n.distance:
                    raw_matches.append(m)

        points_a = np.array([keypoints_a[m.queryIdx].pt for m in raw_matches], dtype=float)
        points_b = np.array([keypoints_b[m.trainIdx].pt for m in raw_matches], dtype=float)
        return points_a, points_b, raw_matches
