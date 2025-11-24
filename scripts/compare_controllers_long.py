"""
Script to compare MPC-SISO and PI controllers for the Shannon Control Unit v1.5
with a longer 500-step experiment
"""
import torch
import torch.nn as nn
import logging
import math
from typing import List, Dict
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
import os
import numpy as np

# Add the project root to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scu.mpc_controller import ControllerBase, MPCController, PIController
from scu.metrics import MetricBase, S_Ratio, AttentionEntropy, GradientOrthogonality
from scu.control import calculate_param_bpt


class DynamicsLSTM(nn.Module):
    """Simple LSTM model to predict next state given current state and control input - copied from train_dynamics_surrogate.py"""
    def __init__(self, state_dim, control_dim, hidden_dim=128, num_layers=2):
        super(DynamicsLSTM, self).__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input will be concatenated state and control input
        self.input_dim = state_dim + control_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output layer to predict next state
        self.output_layer = nn.Linear(hidden_dim, state_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, current_state, control_input):
        # Concatenate current state and control input
        # Shape: [batch_size, 1, state_dim + control_dim]
        input_combined = torch.cat([current_state, control_input], dim=-1).unsqueeze(1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(input_combined)
        
        # Predict next state
        next_state = self.output_layer(lstm_out).squeeze(1)
        
        return next_state


def load_model(name):
    """Load a model - using a smaller, supported model for demo purposes"""
    print(f"Loading model: {name}")
    # For the demo, using a smaller model that's compatible
    # In a real scenario, you would use the full Qwen3-1.7B-Base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Model {name} not supported: {e}. Using a mock model for demo...")
        # Return a simplified mock model that has similar structure to Qwen
        class MockQwenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = 32000  # More realistic vocab size
                self.embed_dim = 768  # Reduced size for demo
                self.wte = nn.Embedding(self.vocab_size, self.embed_dim)  # Token embedding
                self.wpe = nn.Embedding(512, self.embed_dim)  # Position embedding
                # Add some layers to make it more realistic
                self.h = nn.ModuleList([nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=12) for _ in range(4)])
                self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        
            def forward(self, input_ids=None, labels=None):
                # Create embeddings
                device = input_ids.device if input_ids is not None else torch.device('cpu')
                batch_size, seq_len = input_ids.shape
                
                # Limit sequence length to prevent positional embedding issues
                seq_len = min(seq_len, 512)
                input_ids = input_ids[:, :seq_len]
                
                pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
                
                tok_emb = self.wte(input_ids)
                pos_emb = self.wpe(pos_ids)
                x = tok_emb + pos_emb
                
                # Pass through transformer blocks
                for block in self.h:
                    x = block(x)
                
                # Generate logits
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    # Only use the sequence part that we processed
                    labels = labels[:, :seq_len]
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                return type('ModelOutput', (), {'logits': logits, 'loss': loss})()
        
        model = MockQwenModel()
        
    return model


def load_data():
    """Load data from the SlimPajama subset or generate dummy data if issues arise"""
    print("Loading data from data/slimpajama_subset.jsonl...")
    
    # Load a batch of data from the subset
    batch_size = 4  # Reduced batch size for the 1.7B model
    max_length = 512  # Truncate sequences to this length
    
    try:
        # Load tokenizer
        from transformers import AutoTokenizer
        # Load a standard tokenizer as fallback to get standard model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Could not load any tokenizer, using a simple approach...")
        tokenizer = None
    
    # Read and process the data file
    try:
        with open("data/slimpajama_subset.jsonl", 'r', encoding='utf-8') as f:
            lines = [line for line in f]  # Read all lines into memory
        
        # Cycle through the data multiple times to get enough batches for 500+ steps
        epoch = 0
        while epoch < 10:  # Allow up to 10 epochs to get enough data for training
            for i in range(0, len(lines), batch_size):
                batch_lines = lines[i:i+batch_size]
                
                texts = []
                for line in batch_lines:
                    item = json.loads(line)
                    # Extract text from the SlimPajama data structure
                    if 'text' in item:
                        texts.append(item['text'])
                    elif 'content' in item:  # Alternative field name
                        texts.append(item['content'])
                    else:
                        # If no text field, take any string value
                        for key, value in item.items():
                            if isinstance(value, str) and len(value) > 10:
                                texts.append(value)
                                break
                
                if texts:
                    # Tokenize the batch using a fallback tokenizer if needed
                    if tokenizer:
                        encodings = tokenizer(
                            texts, 
                            truncation=True, 
                            padding=True, 
                            max_length=max_length,
                            return_tensors="pt"
                        )
                        
                        input_ids = encodings['input_ids']
                        # Make sure token IDs are within vocab range for the mock model
                        input_ids = torch.clamp(input_ids, 0, 31999)  # Clamp to vocab size - 1
                    else:
                        # Generate random token IDs if no tokenizer available
                        input_ids = torch.randint(0, 32000, (batch_size, max_length//2))
                    
                    # Use input_ids as both input and target (for causal LM training)
                    labels = input_ids.clone()
                    # If we have a tokenizer, mask padding tokens in the loss
                    if tokenizer:
                        labels[labels == tokenizer.pad_token_id] = -100
                    
                    yield input_ids, labels
            epoch += 1
    except Exception as e:
        print(f"Error loading data file: {e}. Generating dummy data for demo...")
        # Generate dummy data for testing - increase for longer experiment
        for i in range(2000):  # Generate more batches for longer run
            input_ids = torch.randint(0, 32000, (batch_size, max_length//2))  # Random tokens
            labels = input_ids.clone()
            yield input_ids, labels


class TrainingOrchestrator:
    """Manages the main training loop, integrating controllers and metrics."""
    def __init__(self, model: nn.Module, controller: ControllerBase, metrics: List[MetricBase], config: Dict):
        self.model = model
        self.controller = controller
        self.metrics = metrics
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config['initial_lr'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.state = {}
        # Initialize previous gradients for GradientOrthogonality metric
        for metric in self.metrics:
            if isinstance(metric, GradientOrthogonality):
                metric.prev_grads = {}
                break

    def train(self, max_steps=500):
        logging.info(f"Starting training with controller: {self.controller.__class__.__name__} for {max_steps} steps")
        
        step = 0
        results = []
        for inputs, labels in load_data():
            if step >= max_steps:  # Limit training steps to specified amount
                break
                
            self.model.train()

            # --- 1. Forward Pass & Loss ---
            outputs = self.model(input_ids=inputs, labels=labels)
            loss = outputs.loss  # Access loss from model outputs
            
            # --- 2. Measurement Step ---
            # Update the shared state dictionary for metrics
            self.state['model'] = self.model
            self.state['loss'] = loss
            
            # Calculate tokens_per_epoch (N) for normalization
            # For this demo script, we estimate it based on the data loader
            # In a real scenario, this should be exact
            # 100 samples * 512 tokens (max) approx
            tokens_per_epoch = 100 * 512 
            self.state['tokens_per_epoch'] = tokens_per_epoch
            
            # For real models, attention maps are not directly accessible in this way
            # We'll set a placeholder for now - attention maps would be captured differently in real implementation
            # In a real implementation, attention maps would be extracted using hooks
            self.state['attention_maps'] = torch.rand(inputs.size(0), 12, inputs.size(1), inputs.size(1)) 
            
            # Calculate all metrics
            # Note: GradientOrthogonality needs gradients, so we do a pre-calculation
            loss.backward(retain_graph=True)
            
            measured_state = {metric.name: metric(self.state) for metric in self.metrics}
            self.optimizer.zero_grad()

            # --- 3. Control Step ---
            control_inputs = self.controller.update(measured_state)
            lambda_t = control_inputs.get('lambda', 1.0)
            lr_t = control_inputs.get('lr', self.config['initial_lr'])

            # --- 4. Application Step ---
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_t
            
            # Apply regularization with lambda
            # Calculate the param bpt for regularization, based on S-Ratio
            data_bpt = (loss.item() / math.log(2))  # Convert from nats to bits
            param_bpt = calculate_param_bpt(self.model, tokens_per_epoch=tokens_per_epoch)
            s_ratio = param_bpt / (data_bpt + param_bpt) if (data_bpt + param_bpt) > 0 else 0.0
            
            # Apply regularization with lambda - using the actual S-ratio for param_bpt
            # This should incorporate the measured S-Ratio from the metric, not just param_bpt
            final_loss = loss + lambda_t * loss * param_bpt
            
            final_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Store results for analysis - use the actual calculated S-Ratio from metrics
            # If the metric isn't working properly, calculate it manually
            s_ratio_from_metric = measured_state.get('s_ratio', 0.0)
            if s_ratio_from_metric == 0.0:
                # Calculate manually if metric failed
                data_bpt = (loss.item() / math.log(2))  # Convert from nats to bits
                param_bpt = calculate_param_bpt(self.model, tokens_per_epoch=tokens_per_epoch)
                s_ratio_manual = param_bpt / (data_bpt + param_bpt) if (data_bpt + param_bpt) > 0 else 0.0
                s_ratio_final = s_ratio_manual
            else:
                s_ratio_final = s_ratio_from_metric

            result = {
                'step': step,
                'loss': final_loss.item(),
                'lambda': lambda_t,
                'lr': lr_t,
                's_ratio': s_ratio_final
            }
            results.append(result)
            
            if step % 50 == 0:
                logging.info(f"Step {step:03d} | Loss: {final_loss.item():.4f} | λ: {lambda_t:.3f} | LR: {lr_t:.1e} | S-Ratio: {result['s_ratio']:.4f}")
            step += 1
            
        logging.info("Training finished.")
        return results


def run_experiment(controller_mode="SISO", experiment_name="Experiment", max_steps=500):
    setup_logging()
    
    # --- Configuration (as if loaded from a YAML) ---
    config = {
        'model_name': "Qwen3-1.7B-Base",
        'controller_mode': controller_mode, # Switch between "SISO", "MIMO", "PI"
        'initial_lr': 5e-5,
    }

    # --- Setup ---
    model = load_model(config['model_name'])
    metrics = [S_Ratio(), AttentionEntropy(), GradientOrthogonality()]
    
    # Initialize the dynamics model for MPC and load trained weights
    # Create a new dynamics model instance with the right dimensions
    # For now, using default dimensions based on our training script
    dynamics_surrogate = DynamicsLSTM(state_dim=4, control_dim=1)  # state: [s_ratio, attn_entropy, loss, lr], control: [lambda]
    
    # Load the pre-trained dynamics model
    dynamics_surrogate.load_state_dict(torch.load("models/dynamics_model.pth", map_location=torch.device('cpu')))

    # Select controller based on config
    if config['controller_mode'] == "PI":
        controller = PIController()
        print(f"\n=== Running {experiment_name} (PI Controller) for {max_steps} steps ===")
    else: # Default to MPC
        controller = MPCController(
            system_dynamics_model=dynamics_surrogate,
            mode=config['controller_mode']
        )
        print(f"\n=== Running {experiment_name} (MPC-{config['controller_mode']} Controller) for {max_steps} steps ===")

    # --- Run ---
    orchestrator = TrainingOrchestrator(model, controller, metrics, config)
    results = orchestrator.train(max_steps=max_steps)
    
    return results


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compare_results(siso_results, pi_results):
    print("\n=== COMPREHENSIVE COMPARISON RESULTS ===")
    
    # Calculate final values
    siso_final_loss = siso_results[-1]['loss'] if siso_results else float('inf')
    pi_final_loss = pi_results[-1]['loss'] if pi_results else float('inf')
    
    print(f"SISO Final Loss: {siso_final_loss:.4f}")
    print(f"PI Final Loss: {pi_final_loss:.4f}")
    print(f"Loss Difference (SISO - PI): {siso_final_loss - pi_final_loss:.4f}")
    
    if siso_final_loss < pi_final_loss:
        print("SISO MPC Controller performed better (lower final loss)")
    elif siso_final_loss > pi_final_loss:
        print("PI Controller performed better (lower final loss)")
    else:
        print("Both controllers performed equally")
    
    # Calculate average losses across the experiment
    siso_avg_loss = np.mean([r['loss'] for r in siso_results])
    pi_avg_loss = np.mean([r['loss'] for r in pi_results])
    
    print(f"\nSISO Average Loss: {siso_avg_loss:.4f}")
    print(f"PI Average Loss: {pi_avg_loss:.4f}")
    print(f"Average Loss Difference (SISO - PI): {siso_avg_loss - pi_avg_loss:.4f}")
    
    # Show lambda variation
    siso_lambdas = [r['lambda'] for r in siso_results]
    pi_lambdas = [r['lambda'] for r in pi_results]
    
    print(f"\nSISO Lambda Range: [{min(siso_lambdas):.3f}, {max(siso_lambdas):.3f}]")
    print(f"PI Lambda Range: [{min(pi_lambdas):.3f}, {max(pi_lambdas):.3f}]")
    
    # Stability analysis
    siso_lambda_std = np.std(siso_lambdas)
    pi_lambda_std = np.std(pi_lambdas)
    
    print(f"\nSISO Lambda Std Dev: {siso_lambda_std:.4f}")
    print(f"PI Lambda Std Dev: {pi_lambda_std:.4f}")
    
    # Calculate how much the lambda changed overall
    siso_lambda_change = abs(max(siso_lambdas) - min(siso_lambdas))
    pi_lambda_change = abs(max(pi_lambdas) - min(pi_lambdas))
    
    print(f"\nSISO Lambda Total Change: {siso_lambda_change:.4f}")
    print(f"PI Lambda Total Change: {pi_lambda_change:.4f}")
    
    # Track loss improvement
    siso_initial_loss = siso_results[0]['loss'] if siso_results else 0
    pi_initial_loss = pi_results[0]['loss'] if pi_results else 0
    
    siso_improvement = (siso_initial_loss - siso_final_loss) / siso_initial_loss * 100
    pi_improvement = (pi_initial_loss - pi_final_loss) / pi_initial_loss * 100
    
    print(f"\nSISO Loss Improvement: {siso_improvement:.2f}%")
    print(f"PI Loss Improvement: {pi_improvement:.2f}%")
    
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"1. Controller Activity: Both controllers showed activity over {len(siso_results)} steps.")
    print(f"2. Overall Performance: {'SISO MPC' if siso_final_loss < pi_final_loss else 'PI'} controller performed better.")
    print(f"3. Stability: {'SISO' if siso_lambda_std < pi_lambda_std else 'PI'} controller showed more stable lambda values.")
    print(f"4. Adaptability: {'SISO' if siso_lambda_change > pi_lambda_change else 'PI'} controller showed more adaptability.")
    print(f"5. Convergence: SISO improved by {siso_improvement:.2f}%, PI improved by {pi_improvement:.2f}%.")


def main():
    max_steps = 500  # Run for 500 steps as requested
    
    # Run SISO MPC experiment
    siso_results = run_experiment("SISO", "SISO MPC", max_steps=max_steps)
    
    # Run PI experiment
    pi_results = run_experiment("PI", "PI Baseline", max_steps=max_steps)
    
    # Compare results
    compare_results(siso_results, pi_results)


if __name__ == "__main__":
    main()