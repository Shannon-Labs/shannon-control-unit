#!/usr/bin/env python3
"""
Cleanup helper for Hugging Face repo to unify ablation CSVs under ablations/.

Deletes duplicate root-level files so README links to ablations/ remain consistent.
"""
from huggingface_hub import HfApi, list_repo_files, whoami

REPO = "hunterbown/shannon-control-unit"
FILES_TO_DELETE = [
    "pi_control.csv",
    "fixed_0.5.csv",
    "fixed_1.0.csv",
    "fixed_2.0.csv",
    "fixed_5.0.csv",
    # Old path for validation results (now lives under results/)
    "3b_validation_results.json",
]

def main():
    api = HfApi()
    user = whoami()
    print(f"Logged in as {user['name']}")
    files = set(list_repo_files(REPO, repo_type="model"))
    print(f"Repo has {len(files)} files")
    to_delete = [f for f in FILES_TO_DELETE if f in files]
    if not to_delete:
        print("Nothing to delete. Repo already clean.")
        return
    print("Deleting:")
    for f in to_delete:
        print(f" - {f}")
        api.delete_file(
            repo_id=REPO,
            path_in_repo=f,
            repo_type="model",
            commit_message="chore: unify ablations under folder; remove root duplicates"
        )
    print("Done.")

if __name__ == "__main__":
    main()

