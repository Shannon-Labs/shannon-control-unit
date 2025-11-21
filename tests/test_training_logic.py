import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from scu_api.config import TrainingConfig
from scu_api.training_engine import TrainingEngine


@pytest.fixture
def mock_accelerator():
    with patch("scu_api.training_engine.Accelerator") as mock:
        acc = mock.return_value
        acc.device = torch.device("cpu")
        acc.prepare.side_effect = lambda m, o, s: (m, o, s)
        acc.unwrap_model.side_effect = lambda m: m
        yield acc


@pytest.fixture
def mock_model_tokenizer():
    with patch("scu_api.training_engine.AutoModelForCausalLM") as mock_model_cls, \
         patch("scu_api.training_engine.AutoTokenizer") as mock_tok_cls, \
         patch("scu_api.training_engine.get_peft_model") as mock_peft:
        
        # Mock Tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        tokenizer.model_max_length = 1024
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        mock_tok_cls.from_pretrained.return_value = tokenizer

        # Mock Model
        model = MagicMock()
        model.config.use_cache = True
        # Mock forward pass output
        output = MagicMock()
        output.loss = torch.tensor(2.5)
        model.return_value = output
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        mock_model_cls.from_pretrained.return_value = model
        
        # Mock PEFT
        peft_model = MagicMock()
        peft_model.print_trainable_parameters = MagicMock()
        peft_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        # Forward pass on peft model
        peft_model.return_value = output
        mock_peft.return_value = peft_model

        yield model, tokenizer


def test_training_engine_run(tmp_path, mock_accelerator, mock_model_tokenizer):
    """Test that TrainingEngine runs a minimal loop without errors."""
    
    # Create dummy data
    data_file = tmp_path / "train.txt"
    data_file.write_text("This is a test sentence for training.\n" * 10)

    config = TrainingConfig(
        base_model="gpt2",
        train_data=str(data_file),
        steps=2,  # Run only 2 steps
        batch_size=1,
        adapter_out=str(tmp_path / "adapter"),
        use_unsloth=False,  # Force standard path
        log_csv=str(tmp_path / "log.csv")
    )

    engine = TrainingEngine(config, job_id="test_job")
    
    # Mock data loading to avoid actual tokenization overhead/errors
    with patch("shannon_control.data.load_texts_from_file", return_value=["text"] * 5), \
         patch("shannon_control.data.tokenize_and_chunk", return_value=[{"input_ids": [1], "attention_mask": [1]}] * 5):
        
        adapter_path = engine.run()

    assert adapter_path.exists()
    assert (adapter_path / "metadata.json").exists()
    assert (tmp_path / "log.csv").exists()
    
    # Verify logs contain expected headers
    with open(tmp_path / "log.csv") as f:
        header = f.readline()
        assert "s_ratio" in header or "S" in header