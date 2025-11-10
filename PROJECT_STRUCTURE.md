# Project Structure

Shannon Control Unit repository organization after Phase 2 cleanup.

## Directory Layout

```
shannon-control-unit/
├── scu/                      # Core SCU implementation
│   ├── __init__.py
│   ├── control.py           # PI controller for adaptive lambda
│   └── data.py              # Data loading utilities
│
├── scripts/                  # Training and evaluation scripts
│   ├── train_scu.py         # Main training script
│   ├── eval_bpt.py          # BPT evaluation
│   ├── run_ablation.py      # Ablation studies
│   ├── reproduce_scu_training.py
│   ├── test_quickstart.py   # Quick validation test
│   └── baselines/           # Baseline comparison scripts
│       └── kl_targeting.py
│
├── viz/                      # Visualization tools
│   ├── plots.py             # Core plotting functions
│   ├── cli.py               # Command-line interface
│   ├── generate_ablation_plots.py
│   └── generate_ablation_analysis.py
│
├── tools/                    # Utility tools
│   ├── hf_sync.py           # HuggingFace synchronization
│   ├── push_to_hf.py        # Model upload
│   └── visual_audit.js      # Visual QA checks
│
├── docs/                     # Documentation
│   ├── README.md            # API reference and guides
│   └── technical/           # Technical deep-dives
│       └── README.md
│
├── examples/                 # Code examples and tutorials
│   └── README.md            # Usage examples
│
├── notebooks/                # Jupyter notebooks
│   └── SCU_Demo.ipynb       # Interactive demo
│
├── web/                      # Landing page (shannonlabs.dev)
│   ├── index.html
│   ├── styles.css
│   ├── js/
│   └── images/
│
├── configs/                  # Configuration files
│   └── default.yaml
│
├── data/                     # Training/validation data
│   └── val.txt              # WikiText validation subset
│
├── results/                  # Validation results
│   ├── 3b_validation_results.json
│   └── audit_results.json
│
├── tests/                    # Unit tests
│   └── test_control.py
│
├── archive/                  # Historical materials
│   ├── README.md
│   └── scu_outreach_kit/    # Early outreach materials
│
├── 1b-scu/                   # 1B model adapter files
├── 3b-scu/                   # 3B model adapter files
├── 3b-fixed/                 # 3B baseline (fixed lambda)
│
├── README.md                 # Main project README
├── README_HF.md              # HuggingFace model card
├── CHANGELOG.md              # Version history
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # AGPL-3.0 license
├── requirements.txt          # Python dependencies
├── vercel.json              # Vercel deployment config
└── .gitignore               # Git ignore rules
```

## Key Files

### Core Implementation
- [`scu/control.py`](scu/control.py) - `SCUController` class with PI control logic
- [`scu/data.py`](scu/data.py) - Data loading and preprocessing

### Training & Evaluation
- [`scripts/train_scu.py`](scripts/train_scu.py) - Complete training pipeline
- [`scripts/eval_bpt.py`](scripts/eval_bpt.py) - BPT metric evaluation
- [`scripts/run_ablation.py`](scripts/run_ablation.py) - Ablation experiments

### Documentation
- [`README.md`](README.md) - Project overview, quick start, results
- [`docs/README.md`](docs/README.md) - Complete API documentation
- [`examples/README.md`](examples/README.md) - Usage examples and tutorials
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - How to contribute

### Model Artifacts
- `1b-scu/` - Llama-3.2-1B + SCU adapter (-6.2% BPT)
- `3b-scu/` - Llama-3.2-3B + SCU adapter (-10.6% BPT)
- `results/` - Validation metrics and analysis

## Organization Principles

### By Function
- **Core code** (`/scu`) - Minimal, well-tested implementation
- **Scripts** (`/scripts`) - Executable training/eval workflows
- **Documentation** (`/docs`, `/examples`) - Learning resources
- **Artifacts** (`/results`, model directories) - Experimental outputs
- **Archive** (`/archive`) - Historical reference, not production

### Clean Separation
- Code vs Documentation vs Web
- Core vs Utilities vs Tools
- Current vs Historical (archive)

### Investor-Ready
- Clear entry points (README, examples)
- Validation results prominently featured
- Professional documentation structure
- Easy reproducibility

## Navigation Tips

**New to SCU?**
1. Start with [main README](README.md)
2. Try [examples](examples/README.md)
3. Read [API docs](docs/README.md)

**Want to contribute?**
1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Check [CHANGELOG.md](CHANGELOG.md) for roadmap
3. Explore [scripts/](scripts/) for implementation patterns

**Researcher/academic?**
1. Review [validation results](results/)
2. Check [technical docs](docs/technical/)
3. Examine training scripts in [scripts/](scripts/)

**Deploying to production?**
1. Review [requirements.txt](requirements.txt)
2. Study [scripts/train_scu.py](scripts/train_scu.py)
3. Check [API reference](docs/README.md)

## Recent Changes

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**Phase 2 (2025-10-10)**: Repository organization and cleanup
- Removed obsolete planning docs
- Archived legacy materials
- Created comprehensive documentation structure
- Added contribution guidelines

**Phase 1 (2025-09)**: Initial validation
- Validated on Llama-3.2-1B and 3B models
- Published to HuggingFace
- Created landing page
