import pytest
import numpy as np
from unittest.mock import Mock, patch

from client.client_app import PanoramaMatcherClient
from client.privacy import validate_client_payload, validate_model_parameters

class TestPrivacyValidation:
    """Test privacy validation functions."""

    def test_validate_model_parameters_valid(self):
        """Test that valid numpy arrays pass validation."""
        params = [np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0], [3.0, 4.0]])]
        validate_model_parameters(params)

    def test_validate_model_parameters_invalid_type(self):
        """Test that non-numpy arrays fail validation."""
        params = [np.array([1.0, 2.0]), "not_an_array"]
        with pytest.raises(ValueError, match="must be a numpy array"):
            validate_model_parameters(params)

    def test_validate_client_payload_disallowed_keys(self):
        """Test that payloads with disallowed keys are rejected."""
        payload = {"image_path": "/some/path.jpg"}
        with pytest.raises(ValueError, match="Privacy violation"):
            validate_client_payload(payload)

    def test_validate_client_payload_bytes(self):
        """Test that byte payloads are rejected."""
        payload = b"raw image data"
        with pytest.raises(ValueError, match="Privacy violation"):
            validate_client_payload(payload)

class TestFederatedClient:
    """Test the federated client implementation."""

    @patch('client.client_app.create_matcher')
    def test_client_initialization(self, mock_create_matcher):
        """Test client initialization."""
        mock_matcher = Mock()
        mock_create_matcher.return_value = mock_matcher

        config = {
            "model": {"name": "loftr"},
            "data": {"data_dir": "/fake/path"},
            "training": {"batch_size": 4}
        }

        client = PanoramaMatcherClient("client_1", config)

        assert client.client_id == "client_1"
        assert client.config == config
        mock_create_matcher.assert_called_once_with({"name": "loftr"})

    @patch('client.client_app.create_matcher')
    def test_get_parameters(self, mock_create_matcher):
        """Test parameter extraction."""
        mock_param = Mock()
        mock_param.detach.return_value.cpu.return_value.numpy.return_value = np.array([1.0, 2.0])

        mock_model = Mock()
        mock_model.parameters.return_value = [mock_param]

        mock_matcher = Mock()
        mock_matcher.model = mock_model
        mock_create_matcher.return_value = mock_matcher

        config = {"model": {"name": "loftr"}}
        client = PanoramaMatcherClient("client_1", config)

        params = client.get_parameters()

        assert len(params) == 1
        assert isinstance(params[0], np.ndarray)
        np.testing.assert_array_equal(params[0], np.array([1.0, 2.0]))

    @patch('client.client_app.create_matcher')
    def test_set_parameters(self, mock_create_matcher):
        """Test parameter setting."""
        mock_param = Mock()
        mock_param.device = "cpu"

        mock_model = Mock()
        mock_model.parameters.return_value = [mock_param]

        mock_matcher = Mock()
        mock_matcher.model = mock_model
        mock_create_matcher.return_value = mock_matcher

        config = {"model": {"name": "loftr"}}
        client = PanoramaMatcherClient("client_1", config)

        new_params = [np.array([3.0, 4.0])]
        client.set_parameters(new_params)

        mock_param.copy_.assert_called_once()
