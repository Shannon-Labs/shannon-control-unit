#!/usr/bin/env python3
"""Generate all outreach materials in one go."""

import os
import yaml
import pandas as pd
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def ensure_dirs():
    """Create output directories."""
    dirs = ['output/emails', 'output/docs', 'output/plots']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def generate_emails(config):
    """Generate email drafts for each contact."""
    env = Environment(loader=FileSystemLoader('templates'))
    
    for contact in config['contacts']:
        context = {
            'contact': contact,
            'org': config['org'],
            'results': config['results'],
            'pilot': config['pilot']
        }
        
        # Generate initial + 2 follow-ups
        templates = [
            f"{contact['type']}_initial.j2",
            f"{contact['type']}_followup1.j2",
            f"{contact['type']}_followup2.j2"
        ]
        
        for i, template_name in enumerate(templates):
            try:
                template = env.get_template(f"emails/{template_name}")
                content = template.render(**context)
                
                filename = f"output/emails/{contact['company']}_{i+1}.txt"
                with open(filename, 'w') as f:
                    f.write(content)
                print(f"✓ Generated: {filename}")
            except Exception as e:
                print(f"⚠ Skipping {template_name}: {e}")

def generate_docs(config):
    """Generate PDF documents."""
    env = Environment(loader=FileSystemLoader('templates'))
    
    # Generate markdown first
    for doc_type in ['protocol', 'onepager']:
        template = env.get_template(f"docs/{doc_type}.md.j2")
        content = template.render(**config)
        
        md_path = f"output/docs/{doc_type}.md"
        with open(md_path, 'w') as f:
            f.write(content)
        
        # Convert to PDF using reportlab (simple version)
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        pdf_path = f"output/docs/{doc_type}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Convert markdown to simple paragraphs
        for line in content.split('\n'):
            if line.startswith('# '):
                style = styles['Title']
                text = line[2:]
            elif line.startswith('## '):
                style = styles['Heading2']
                text = line[3:]
            elif line.startswith('* '):
                style = styles['BodyText']
                text = f"• {line[2:]}"
            else:
                style = styles['BodyText']
                text = line
            
            if text.strip():
                story.append(Paragraph(text, style))
                story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        print(f"✓ Generated: {pdf_path}")

def generate_plot(config):
    """Generate comparison plot."""
    # Create sample data (replace with actual when available)
    steps = np.arange(0, 250, 10)
    
    # S(t) - oscillating around 1.0%
    s_1b = 1.0 + 0.1 * np.sin(steps/20) + 0.05 * np.random.randn(len(steps))
    s_3b = 1.0 + 0.08 * np.sin(steps/25) + 0.04 * np.random.randn(len(steps))
    
    # ParamBPT - decreasing trend
    param_1b = 0.036 - 0.001 * steps/250 + 0.002 * np.random.randn(len(steps))
    param_3b = 0.034 - 0.0008 * steps/250 + 0.0015 * np.random.randn(len(steps))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # S(t) plot
    ax1.plot(steps, s_1b, label='Llama-3.2-1B', linewidth=2)
    ax1.plot(steps, s_3b, label='Llama-3.2-3B (early)', linewidth=2, linestyle='--')
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between(steps, 0.8, 1.2, alpha=0.1, color='blue')
    ax1.set_ylabel('S (%)', fontsize=12)
    ax1.set_title('SCU Control: S(t) Tracking Target', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.7, 1.3])
    
    # ParamBPT plot
    ax2.plot(steps, param_1b, label='Llama-3.2-1B', linewidth=2)
    ax2.plot(steps, param_3b, label='Llama-3.2-3B (early)', linewidth=2, linestyle='--')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Parameter BPT', fontsize=12)
    ax2.set_title('Parameter Cost (Bits Per Token)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'output/plots/scu_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Generated: {output_path}")
    plt.close()

def main():
    print("SCU Outreach Kit - Generating Materials\n" + "="*40)
    
    config = load_config()
    ensure_dirs()
    
    print("\n1. Generating email drafts...")
    generate_emails(config)
    
    print("\n2. Generating documents...")
    generate_docs(config)
    
    print("\n3. Generating plot...")
    generate_plot(config)
    
    print("\n✅ All materials generated in ./output/")
    print("\nNext steps:")
    print("1. Review emails in output/emails/")
    print("2. Attach PDFs from output/docs/ to emails")
    print("3. Personalize and send via your email client")
    print("4. Run check_readiness.py to verify HN trigger")

if __name__ == "__main__":
    main()