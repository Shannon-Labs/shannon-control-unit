import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import os
from torch.utils.data import DataLoader, Dataset


class DynamicsDataset(Dataset):
    """Dataset class for training the dynamics surrogate model"""
    def __init__(self, sequences, control_inputs):
        """
        Args:
            sequences: List of state sequences [batch_size, seq_len, state_dim]
            control_inputs: List of control inputs corresponding to each transition
        """
        self.sequences = sequences
        self.control_inputs = control_inputs
        
    def __len__(self):
        return len(self.sequences) - 1  # We need current and next states
    
    def __getitem__(self, idx):
        current_state = self.sequences[idx]
        next_state = self.sequences[idx + 1]
        control_input = self.control_inputs[idx]
        
        return {
            'current_state': torch.tensor(current_state, dtype=torch.float32),
            'control_input': torch.tensor(control_input, dtype=torch.float32),
            'next_state': torch.tensor(next_state, dtype=torch.float32)
        }


class DynamicsLSTM(nn.Module):
    """Simple LSTM model to predict next state given current state and control input"""
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


def simulate_training_run(model, tokenizer, dataset_path, num_steps=100):
    """
    Simulate a training run of the Qwen3-1.7B-Base model with fixed lambda
    Log state vector (S-Ratio, Attention Entropy, etc.) and control input at each step
    """
    print("Starting baseline training run simulation to collect dynamics data...")
    
    # Load a small portion of the dataset for simulation
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = [next(f) for _ in range(5000)]  # Read 5000 samples for the simulation
    
    # Parse the data
    parsed_data = [json.loads(line) for line in lines]
    
    # Initialize lists to store states and control inputs
    states_log = []
    controls_log = []
    
    # Fixed lambda for baseline run
    lambda_fixed = 0.01
    
    # Simulate training steps
    for step in range(min(num_steps, len(parsed_data))):
        # Simulate extracting state features from the model during training
        # In a real scenario, this would involve actual training steps
        s_ratio = np.random.uniform(0.005, 0.02)  # Simulated S-ratio
        attention_entropy = np.random.uniform(2.0, 8.0)  # Simulated attention entropy
        loss = max(1e-6, 5.0 - step * 0.01)  # Simulated decreasing loss
        learning_rate = 5e-5  # Fixed learning rate
        
        # Create state vector: [s_ratio, attention_entropy, loss, learning_rate]
        state = [s_ratio, attention_entropy, loss, learning_rate]
        control = [lambda_fixed]
        
        states_log.append(state.copy())
        controls_log.append(control.copy())
        
        if step % 20 == 0:
            print(f"Step {step}: State={state}, Control={control}")
    
    print(f"Collected {len(states_log)} state-control pairs")
    return np.array(states_log), np.array(controls_log)


def train_dynamics_model(states_log, controls_log):
    """
    Train the LSTM dynamics model to predict next state given current state and control input
    """
    print("Training dynamics surrogate model...")
    
    # Prepare the dataset
    # We need to create sequences where each sample has (current_state, control_input) -> next_state
    state_dim = states_log.shape[1]
    control_dim = controls_log.shape[1]
    
    # Convert logs to the required format for the LSTM
    sequences = torch.tensor(states_log, dtype=torch.float32)
    controls = torch.tensor(controls_log, dtype=torch.float32)
    
    # Create dataset
    dataset = []
    for i in range(len(sequences) - 1):
        current_state = sequences[i]
        control_input = controls[i]
        next_state = sequences[i + 1]
        dataset.append((current_state, control_input, next_state))
    
    # Split into train and validation sets
    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Create model
    model = DynamicsLSTM(state_dim=state_dim, control_dim=control_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = 50
    batch_size = 32
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Simple batch creation
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            current_states = torch.stack([item[0] for item in batch])
            control_inputs = torch.stack([item[1] for item in batch])
            next_states = torch.stack([item[2] for item in batch])
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_next_states = model(current_states, control_inputs)
            loss = criterion(pred_next_states, next_states)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_current_states = torch.stack([item[0] for item in val_data])
            val_control_inputs = torch.stack([item[1] for item in val_data])
            val_next_states = torch.stack([item[2] for item in val_data])
            
            val_pred_next_states = model(val_current_states, val_control_inputs)
            val_loss = criterion(val_pred_next_states, val_next_states)
        
        model.train()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {epoch_loss/len(train_data):.6f}, Val Loss: {val_loss:.6f}")
    
    print("Dynamics model training completed!")
    return model


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs("models", exist_ok=True)
    
    # Simulate training run to collect dynamics data
    states_log, controls_log = simulate_training_run(None, None, "data/slimpajama_subset.jsonl", num_steps=1000)
    
    # Save the logs to a CSV file for analysis
    dynamics_df = pd.DataFrame(
        np.concatenate([states_log, controls_log], axis=1),
        columns=['s_ratio', 'attention_entropy', 'loss', 'learning_rate', 'lambda']
    )
    dynamics_df.to_csv("dynamics_log.csv", index=False)
    print("Dynamics log saved to dynamics_log.csv")
    
    # Train the dynamics model
    dynamics_model = train_dynamics_model(states_log, controls_log)
    
    # Save the trained model
    model_path = "models/dynamics_model.pth"
    torch.save(dynamics_model.state_dict(), model_path)
    print(f"Dynamics model saved to {model_path}")
    

if __name__ == "__main__":
    main()