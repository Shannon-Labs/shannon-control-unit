# Changelog

All notable changes to the Shannon Control Unit project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 2 Planning
- External validation at 7B+ scale
- Multi-seed statistical validation with confidence intervals
- Baseline comparisons: KL-targeting, trust-region methods
- Downstream task evaluation (MMLU, GSM8K, etc.)
- Scaling law investigation for optimal S* prediction

---

## [0.2.0] - 2025-10-10

### Added - Repository Organization Phase
- Created comprehensive documentation structure in `/docs`
- Added API reference documentation
- Added `/examples` directory with usage patterns
- Created `/archive` for historical materials
- Added `.gitignore` for repository hygiene
- Added `CONTRIBUTING.md` for next-phase collaborators
- Added this CHANGELOG to track evolution

### Changed
- Moved `scu_outreach_kit` to `/archive`
- Organized documentation into logical hierarchy
- Cleaned up temporary planning documents

### Removed
- Obsolete planning documents (FIND_BEAUTIFUL_VERSION.md, UPDATE_PLAN.md, PHOTO_NEEDED.md)
- Duplicate notebook from `/web` directory
- Legacy visualization code from `/viz/legacy`
- Temporary image processing scripts
- Duplicate `datasets copy/` directory

---

## [0.1.0] - 2025-09-XX

### Added - Initial Validation Phase
- Validated SCU on Llama-3.2-1B: **-6.2% BPT improvement**
- Validated SCU on Llama-3.2-3B: **-10.6% BPT improvement**
- Published model weights on HuggingFace: `hunterbown/shannon-control-unit`
- Created validation dataset (WikiText-103 subset, ~512k tokens)
- Implemented PI controller for adaptive lambda regulation
- Added evaluation scripts (`eval_bpt.py`)
- Created ablation study comparing fixed vs adaptive lambda
- Added Jupyter notebook demo (`SCU_Demo.ipynb`)
- Deployed landing page at shannonlabs.dev

### Core Implementation
- Implemented `SCUController` class with PI control
- Added `S` ratio tracking (ParamBPT / Total BPT)
- Implemented lambda bounding for numerical stability
- Created data utilities for WikiText processing
- Added training scripts with LoRA support

### Documentation
- Created main README with cruise control analogy
- Added HuggingFace model card (README_HF.md)
- Documented validation methodology
- Listed limitations and threats to validity
- Added licensing information (AGPL-3.0 + Commercial)

---

## [0.0.1] - 2025-08-XX

### Initial Concept
- Theoretical framework for MDL-based regularization
- PI controller design for adaptive lambda
- Initial experiments validating control approach
- Patent provisional filing (September 2025)

---

## Next Phase Goals

### Technical Validation
- [ ] Scale validation to 7B+ models
- [ ] Multi-seed runs with statistical significance testing
- [ ] Compare against state-of-the-art adaptive methods
- [ ] Evaluate on downstream benchmarks
- [ ] Test on diverse domains beyond WikiText

### Research & Publication
- [ ] Complete academic paper
- [ ] Derive scaling laws for optimal S*
- [ ] Formal convergence proofs
- [ ] Community peer review

### Engineering
- [ ] Full-parameter training support (beyond LoRA)
- [ ] Multi-GPU/distributed training optimization
- [ ] Integration with popular training frameworks
- [ ] Hyperparameter auto-tuning
- [ ] Real-time monitoring dashboard

### Community
- [ ] External validation partnerships
- [ ] Open-source contributions
- [ ] Tutorial videos and workshops
- [ ] Research collaborations

---

**Legend:**
- `Added` - New features
- `Changed` - Changes to existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security improvements
