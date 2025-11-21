import os
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_NAME = "HuggingFaceFW/finewiki"
OUTPUT_FILE = "data/train_v2.txt"
TARGET_SIZE_MB = 500  # Target size in MB (increased from 100 for natural S=1%)

def download_finewiki_subset():
    print(f"Downloading subset of {DATASET_NAME} to {OUTPUT_FILE}...")
    print(f"Target size: {TARGET_SIZE_MB} MB")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Load dataset in streaming mode
    try:
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying 'HuggingFaceFW/fineweb-edu' as fallback...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    current_size = 0
    target_bytes = TARGET_SIZE_MB * 1024 * 1024
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, item in tqdm(enumerate(dataset)):
            text = item.get("text", "")
            if not text:
                continue
                
            # Add some formatting if needed, or just raw text
            f.write(text + "\n\n")
            
            # Check size
            current_size += len(text.encode("utf-8"))
            if current_size >= target_bytes:
                break
                
    print(f"✅ Download complete! Saved {current_size / 1024 / 1024:.2f} MB to {OUTPUT_FILE}")

if __name__ == "__main__":
    download_finewiki_subset()
