#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

def add_background_to_photo(output_path):
    """Create an architectural-style background with grid"""
    
    # Create a canvas (assuming typical architectural photo dimensions)
    width, height = 1200, 800
    
    # Create gradient background
    gradient = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(gradient)
    
    # Create radial gradient effect with architectural blue-gray tones
    center_x = width // 2
    center_y = height // 2
    max_radius = max(width, height)
    
    for i in range(max_radius, 0, -3):
        ratio = i / max_radius
        # Gradient from deep architectural blue to light gray
        r = int(45 + (245 - 45) * ratio)  # 45 to 245
        g = int(55 + (248 - 55) * ratio)  # 55 to 248  
        b = int(75 + (252 - 75) * ratio)  # 75 to 252
        color = (r, g, b)
        
        draw.ellipse([center_x - i, center_y - i, center_x + i, center_y + i], fill=color)
    
    # Add architectural grid pattern
    grid_spacing = 40
    
    # Create grid overlay with transparency
    grid_img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    grid_draw = ImageDraw.Draw(grid_img)
    
    # Draw main grid lines (stronger)
    for x in range(0, width, grid_spacing):
        alpha = 40 if x % (grid_spacing * 2) == 0 else 25
        grid_draw.line([(x, 0), (x, height)], fill=(200, 210, 220, alpha), width=1)
    
    for y in range(0, height, grid_spacing):
        alpha = 40 if y % (grid_spacing * 2) == 0 else 25
        grid_draw.line([(0, y), (width, y)], fill=(200, 210, 220, alpha), width=1)
    
    # Add perspective grid lines for depth
    vanishing_x = width // 2
    vanishing_y = height // 3
    
    for i in range(-10, 11, 2):
        start_x = width // 2 + i * 50
        grid_draw.line([(start_x, height), (vanishing_x, vanishing_y)], 
                      fill=(180, 190, 210, 20), width=1)
    
    # Composite grid onto gradient
    gradient = Image.alpha_composite(gradient.convert('RGBA'), grid_img)
    
    # Add architectural elements - abstract building silhouettes
    silhouette = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    sil_draw = ImageDraw.Draw(silhouette)
    
    # Left building silhouette
    sil_draw.rectangle([0, height-300, 150, height], fill=(30, 40, 55, 30))
    sil_draw.rectangle([50, height-400, 200, height], fill=(35, 45, 60, 25))
    
    # Right building silhouette  
    sil_draw.rectangle([width-150, height-350, width, height], fill=(30, 40, 55, 30))
    sil_draw.rectangle([width-200, height-450, width-50, height], fill=(35, 45, 60, 25))
    
    gradient = Image.alpha_composite(gradient, silhouette)
    
    # Add subtle vignette for focus
    vignette = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    vignette_draw = ImageDraw.Draw(vignette)
    
    for i in range(50):
        alpha = int(i * 1.5)
        offset = i * 8
        vignette_draw.ellipse([offset, offset, width - offset, height - offset],
                             outline=(0, 10, 30, alpha))
    
    gradient = Image.alpha_composite(gradient, vignette)
    
    # Add central highlight area where photo would go
    highlight = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    highlight_draw = ImageDraw.Draw(highlight)
    
    # Create soft spotlight effect in center
    for i in range(100, 0, -2):
        alpha = int(5 * (100 - i) / 100)
        size_x = width // 3 + i * 2
        size_y = height // 3 + i * 2
        highlight_draw.ellipse([center_x - size_x, center_y - size_y, 
                               center_x + size_x, center_y + size_y],
                              fill=(255, 255, 255, alpha))
    
    gradient = Image.alpha_composite(gradient, highlight)
    
    # Convert to RGB and save
    gradient = gradient.convert('RGB')
    
    # Apply final enhancements
    enhancer = ImageEnhance.Contrast(gradient)
    gradient = enhancer.enhance(1.1)
    
    gradient.save(output_path, quality=95)
    print(f"✓ Created architectural background with grid: {output_path}")

# Create the enhanced background
if __name__ == "__main__":
    add_background_to_photo("architectural_bg_with_grid.jpg")
    print("\n✅ Architectural background created successfully!")