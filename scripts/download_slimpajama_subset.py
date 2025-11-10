from datasets import load_dataset
import os

def download_slimpajama_subset():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load the dataset (this will stream)
    print("Loading SlimPajama dataset...")
    dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    
    # Take a subset (e.g., the first 1 million samples)
    print("Creating subset of 1 million samples...")
    subset = dataset.take(1000000)
    
    # Save to a local file for training
    print("Saving subset to data/slimpajama_subset.jsonl...")
    with open("data/slimpajama_subset.jsonl", "w", encoding="utf-8") as f:
        for i, item in enumerate(subset):
            # Write each item as a JSON line
            import json
            f.write(json.dumps(item) + "\n")
            
            # Print progress every 100,000 samples
            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1} samples...")
    
    print("SlimPajama subset saved successfully!")

if __name__ == "__main__":
    download_slimpajama_subset()