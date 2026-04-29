import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm


def _get_lang(item):
    lang = item.get("in_language") or item.get("language") or item.get("lang")
    if lang:
        return str(lang).lower()
    wikiname = item.get("wikiname")
    if isinstance(wikiname, str) and wikiname.endswith("wiki"):
        return wikiname[:-4].lower()
    return None


def _write_sample(fh, text, jsonl):
    if jsonl:
        fh.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
    else:
        fh.write(text + "\n\n")


def download_finewiki_subset(args):
    print(f"Downloading subset of {args.dataset} to {args.output}...")
    print(f"Target size: {args.target_mb} MB")
    print(f"Language filter: {args.lang or 'none'}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying 'HuggingFaceFW/fineweb-edu' as fallback...")
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    if args.shuffle:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    target_bytes = int(args.target_mb * 1024 * 1024)
    current_size = 0
    sample_count = 0
    jsonl = args.output.endswith(".jsonl")

    with open(args.output, "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            text = item.get(args.text_field, "")
            if not text:
                continue
            if args.lang:
                item_lang = _get_lang(item)
                if item_lang != args.lang.lower():
                    continue

            _write_sample(f, text, jsonl)
            current_size += len(text.encode("utf-8"))
            sample_count += 1

            if args.max_samples and sample_count >= args.max_samples:
                break
            if current_size >= target_bytes:
                break

    print(
        f"Done. Saved {current_size / 1024 / 1024:.2f} MB "
        f"({sample_count} samples) to {args.output}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Download a FineWiki subset")
    parser.add_argument("--dataset", default="HuggingFaceFW/finewiki")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/finewiki_en.jsonl")
    parser.add_argument("--target-mb", type=float, default=500)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    download_finewiki_subset(parse_args())
