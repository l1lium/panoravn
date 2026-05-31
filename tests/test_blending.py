import numpy as np
from core.blending import alpha_blend, blend_panorama

def test_alpha_blend_overlap():
    image_a = np.zeros((5, 5, 3), dtype=np.uint8)
    image_b = np.zeros((5, 5, 3), dtype=np.uint8)
    image_a[:] = [255, 0, 0]
    image_b[:] = [0, 255, 0]
    mask = np.ones((5, 5), dtype=np.uint8)
    blended = alpha_blend(image_a, image_b, mask, mask, alpha=0.5)
    assert blended.shape == image_a.shape
    assert blended[0, 0, 0] in {127, 128}
    assert blended[0, 0, 1] in {127, 128}

def test_blend_panorama_alpha_method():
    image_a = np.zeros((3, 3, 3), dtype=np.uint8)
    image_b = np.zeros((3, 3, 3), dtype=np.uint8)
    image_a[0:2, :] = [255, 0, 0]
    image_b[1:3, :] = [0, 255, 0]
    config = {"blending": {"method": "alpha", "alpha": 0.5}}
    pano = blend_panorama([image_a, image_b], config)
    assert pano.shape == image_a.shape
    assert pano[1, 1, 0] in {127, 128}
    assert pano[1, 1, 1] in {127, 128}
