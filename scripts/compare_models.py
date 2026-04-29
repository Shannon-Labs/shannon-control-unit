import mlx.core as mx
from mlx_lm import load, generate
import time

prompts = [
    "What is the capital of France?",
    "Write a Python function to calculate the Fibonacci sequence.",
    "Explain the concept of entropy in simple terms.",
    "What are the benefits of regular exercise?"
]

def run_eval(name, model, tokenizer, prompts):
    print(f"\n{'='*20} {name} {'='*20}")
    results = []
    for p in prompts:
        print(f"\n[Prompt]: {p}")
        start = time.time()
        
        messages = [{"role": "user", "content": p}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                prompt = p
        else:
            prompt = p

        response = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        end = time.time()
        print(f"[Response ({end-start:.2f}s)]:\n{response.strip()}")
        results.append(response.strip())
    return results

print("Loading Base Model: mlx-community/Olmo-3-7B-Instruct-4bit")
model_base, tokenizer_base = load("mlx-community/Olmo-3-7B-Instruct-4bit")
run_eval("Base Model", model_base, tokenizer_base, prompts)

print("\n\nLoading SCU Adapter: adapters/olmo3_7b_fineweb_scu_full")
# Now that adapter_config.json is fixed, this should work
model_scu, tokenizer_scu = load("mlx-community/Olmo-3-7B-Instruct-4bit", adapter_path="adapters/olmo3_7b_fineweb_scu_full")
run_eval("SCU Adapter", model_scu, tokenizer_scu, prompts)
