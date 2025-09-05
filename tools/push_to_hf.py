from huggingface_hub import HfApi
from pathlib import Path
import textwrap

REPO = "hunterbown/shannon-control-unit"
api = HfApi()

# Upload notebook
api.upload_file(
    path_or_fileobj="notebooks/SCU_Demo.ipynb",
    repo_id=REPO,
    path_in_repo="notebooks/SCU_Demo.ipynb",
    repo_type="model",
    commit_message="Add Colab-ready SCU demo notebook"
)

# Upload figures folder (ensure exists)
fig_dir = Path("assets/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
api.upload_folder(
    folder_path=str(fig_dir),
    repo_id=REPO,
    path_in_repo="assets/figures",
    repo_type="model",
    commit_message="Add control figures (S curve, lambda curve)"
)

# Update model card README with badge, validation, images, and licensing
readme = textwrap.dedent(
    f"""
    # Shannon Control Unit (SCU)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hmbown/shannon-control-unit/blob/main/notebooks/SCU_Demo.ipynb)

    ## Validation
    Base 3.920 BPT (ppl 15.14) → SCU 3.676 (ppl 12.78), Δ −6.2% BPT ≈ −15.6% ppl.

    ![S curve](assets/figures/s_curve.png)
    ![Lambda curve](assets/figures/lambda_curve.png)

    ## Licensing & IP
    - Adapters/models: Meta Llama 3.2 Community License
    - SCU training code: AGPL-3.0 (research/academia). Commercial licenses available.
    - U.S. patent pending (provisional filed September 2025)
    """
)

tmp_md = Path("tools/HF_README.md")
tmp_md.write_text(readme)
api.upload_file(
    path_or_fileobj=str(tmp_md),
    repo_id=REPO,
    path_in_repo="README.md",
    repo_type="model",
    commit_message="Update model card: Colab badge, validation, figures, AGPL/commercial licensing"
)

print("Uploaded notebook, figures, and updated model card to:", REPO)
