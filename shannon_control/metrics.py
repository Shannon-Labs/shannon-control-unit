"""
Pluggable metric modules for measuring system state during training.
Following the expert-recommended software architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Import base utilities (assuming they exist in scu.control or a new utils file)
from .control import calculate_data_bpt, calculate_param_bpt, calculate_s_ratio

class MetricBase(nn.Module):
    """Abstract base class for all metric measurement modules."""
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, orchestrator_state: Dict) -> float:
        """
        Calculates the metric based on the current state of the TrainingOrchestrator.
        
        Args:
            orchestrator_state (Dict): A dictionary containing all relevant training
                                     information (model, loss, grads, etc.).
        
        Returns:
            float: The calculated metric value.
        """
        raise NotImplementedError

class S_Ratio(MetricBase):
    """Measures the Information Ratio (S-Ratio)."""
    def __init__(self):
        super().__init__("s_ratio")

    def forward(self, orchestrator_state: Dict) -> float:
        loss = orchestrator_state.get("loss")
        model = orchestrator_state.get("model")
        if loss is None or model is None:
            return 0.0
        
        # tokens_per_epoch is now REQUIRED for correct S-ratio calculation
        tokens_per_epoch = orchestrator_state.get("tokens_per_epoch")
        if tokens_per_epoch is None:
            raise ValueError("tokens_per_epoch must be provided in orchestrator_state for S_Ratio metric")
            
        param_bpt = calculate_param_bpt(model, tokens_per_epoch=tokens_per_epoch)
        return calculate_s_ratio(data_bpt, param_bpt)

class AttentionEntropy(MetricBase):
    """Measures the average Shannon entropy of the model's attention distributions."""
    def __init__(self):
        super().__init__("attn_entropy")

    def forward(self, orchestrator_state: Dict) -> float:
        attention_maps = orchestrator_state.get("attention_maps")
        if attention_maps is None:
            return 0.0
        
        # Add a small epsilon to prevent log(0) for numerical stability
        entropy = -torch.sum(attention_maps * torch.log(attention_maps + 1e-9), dim=-1)
        return torch.mean(entropy).item()

class GradientOrthogonality(MetricBase):
    """Measures the cosine similarity between the current and previous gradients."""
    def __init__(self):
        super().__init__("grad_ortho")
        self.prev_grads = {}

    def forward(self, orchestrator_state: Dict) -> float:
        model = orchestrator_state.get("model")
        if model is None or not hasattr(model, 'parameters'):
            return 0.0

        current_grads = {
            name: p.grad.view(-1).detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None and p.requires_grad
        }

        if not self.prev_grads:
            self.prev_grads = current_grads
            return 0.0

        total_cos_sim = 0.0
        num_params = 0
        for name, grad in current_grads.items():
            if name in self.prev_grads:
                prev_grad_device = self.prev_grads[name].to(grad.device)
                total_cos_sim += F.cosine_similarity(grad.unsqueeze(0), prev_grad_device.unsqueeze(0)).item()
                num_params += 1
        
        self.prev_grads = current_grads
        return total_cos_sim / num_params if num_params > 0 else 0.0
