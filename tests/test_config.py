from core.io import load_config

def test_load_baseline_config():
    config = load_config("configs/baseline.yaml")
    assert isinstance(config, dict)
    assert config["matcher"]["name"] in {"sift", "orb"}
    assert config["paths"]["output_dir"] == "outputs"
    assert config["image"]["max_width"] > 0
    assert config["geometry"]["ransac_threshold"] > 0
