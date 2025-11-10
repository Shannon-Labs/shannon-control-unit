# SCU Outreach Kit

Quick toolkit to prepare materials for hyperscaler outreach.

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Generate all materials
python generate_materials.py

# Check what's ready
python check_readiness.py
```

## What This Creates

1. **Email Templates** → `output/emails/`
   - Initial outreach + 2 follow-ups per contact
   - Ready to copy into your email client

2. **PDF Documents** → `output/docs/`
   - 2-page pilot protocol
   - 1-page summary

3. **Plot PNG** → `output/plots/`
   - Combined S(t) and ParamBPT visualization

4. **HN Readiness Check** → Console output
   - GO/NO-GO based on your criteria

## Customization

Edit `config.yaml` to update:
- Your contact list
- Email preferences
- Document variables
- HN trigger conditions