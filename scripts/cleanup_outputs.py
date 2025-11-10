#!/usr/bin/env python3
"""
Cleanup utility for SCU training output directories

This script helps manage model outputs, logs, and temporary files
generated during SCU training runs.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Set


def get_output_directories() -> List[Path]:
    """Find all output directories in the project"""
    project_root = Path(__file__).parent.parent
    output_dirs = []
    
    # Common output directory patterns
    patterns = [
        "*output*/",
        "*output_test*/",
        "*output_safe*/",
        "*/outputs/",
        "*/lightning_logs/",
        "*/checkpoints/",
        "*/wandb/",
        "*/runs/",
        "*/saved_models/",
        "*/logs/",
        "*/.cache/",
        "*/tmp/",
        "*/temp/",
    ]
    
    for pattern in patterns:
        output_dirs.extend(project_root.glob(pattern))
    
    # Also check for specific model output directories
    specific_dirs = [
        "granite_*_scu_output*",
        "qwen*_scu_output*",
        "simple_scu_*_output*",
        "fixed_scu_*_output*",
        "cpu_scu_*_output*",
        "micro_ultra_*_output*",
    ]
    
    for pattern in specific_dirs:
        output_dirs.extend(project_root.glob(pattern))
    
    # Remove duplicates and non-existent paths
    output_dirs = list(set([d for d in output_dirs if d.exists() and d.is_dir()]))
    output_dirs.sort()
    
    return output_dirs


def get_cache_directories() -> List[Path]:
    """Find cache directories"""
    project_root = Path(__file__).parent.parent
    cache_dirs = []
    
    cache_patterns = [
        ".hf_cache/",
        ".huggingface/",
        "*/.cache/",
        "*/__pycache__/",
        "*/.pytest_cache/",
        "*/node_modules/",
    ]
    
    for pattern in cache_patterns:
        cache_dirs.extend(project_root.glob(pattern))
    
    # Remove duplicates and non-existent paths
    cache_dirs = list(set([d for d in cache_dirs if d.exists() and d.is_dir()]))
    cache_dirs.sort()
    
    return cache_dirs


def get_large_files(min_size_mb: int = 100) -> List[Path]:
    """Find large files that might be model checkpoints or data"""
    project_root = Path(__file__).parent.parent
    large_files = []
    
    # File patterns to check
    file_patterns = [
        "*.pt",
        "*.pth", 
        "*.ckpt",
        "*.safetensors",
        "*.bin",
        "*.csv",
        "*.log",
        "*.jsonl",
    ]
    
    min_bytes = min_size_mb * 1024 * 1024
    
    for pattern in file_patterns:
        for file_path in project_root.rglob(pattern):
            if file_path.exists() and file_path.is_file():
                try:
                    if file_path.stat().st_size > min_bytes:
                        large_files.append(file_path)
                except (OSError, PermissionError):
                    continue
    
    large_files.sort()
    return large_files


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def analyze_directory(directory: Path) -> dict:
    """Analyze directory size and contents"""
    total_size = 0
    file_count = 0
    dir_count = 0
    
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                try:
                    total_size += item.stat().st_size
                    file_count += 1
                except (OSError, PermissionError):
                    continue
            elif item.is_dir():
                dir_count += 1
    except (OSError, PermissionError):
        pass
    
    return {
        'size': total_size,
        'files': file_count,
        'dirs': dir_count
    }


def interactive_cleanup():
    """Interactive cleanup mode"""
    print("🔍 SCU Training Output Cleanup Utility")
    print("=" * 50)
    
    # Find output directories
    output_dirs = get_output_directories()
    cache_dirs = get_cache_directories()
    large_files = get_large_files(min_size_mb=50)
    
    if not output_dirs and not cache_dirs and not large_files:
        print("✨ No cleanup needed - no output directories or large files found!")
        return
    
    items_to_clean = []
    
    # Show output directories
    if output_dirs:
        print(f"\n📁 Found {len(output_dirs)} output directories:")
        for i, directory in enumerate(output_dirs, 1):
            analysis = analyze_directory(directory)
            rel_path = directory.relative_to(Path.cwd())
            print(f"  {i}. {rel_path}")
            print(f"     Size: {format_size(analysis['size'])}, "
                  f"Files: {analysis['files']}, Subdirs: {analysis['dirs']}")
            items_to_clean.append((directory, analysis['size']))
    
    # Show cache directories
    if cache_dirs:
        print(f"\n🗃️  Found {len(cache_dirs)} cache directories:")
        for i, directory in enumerate(cache_dirs, 1):
            analysis = analyze_directory(directory)
            rel_path = directory.relative_to(Path.cwd())
            print(f"  {i}. {rel_path}")
            print(f"     Size: {format_size(analysis['size'])}")
            items_to_clean.append((directory, analysis['size']))
    
    # Show large files
    if large_files:
        print(f"\n📄 Found {len(large_files)} large files:")
        for i, file_path in enumerate(large_files, 1):
            try:
                size = file_path.stat().st_size
                rel_path = file_path.relative_to(Path.cwd())
                print(f"  {i}. {rel_path} ({format_size(size)})")
                items_to_clean.append((file_path, size))
            except (OSError, PermissionError):
                continue
    
    # Calculate total size
    total_size = sum(size for _, size in items_to_clean)
    print(f"\n💾 Total space that can be freed: {format_size(total_size)}")
    
    # Ask for confirmation
    print("\n⚠️  This will permanently delete the selected items!")
    response = input("Do you want to proceed with cleanup? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("❌ Cleanup cancelled.")
        return
    
    # Perform cleanup
    print("\n🗑️  Performing cleanup...")
    freed_space = 0
    
    for item, size in items_to_clean:
        try:
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Deleted directory: {item.relative_to(Path.cwd())}")
            elif item.is_file():
                item.unlink()
                print(f"  ✓ Deleted file: {item.relative_to(Path.cwd())}")
            freed_space += size
        except (OSError, PermissionError) as e:
            print(f"  ✗ Failed to delete {item.relative_to(Path.cwd())}: {e}")
    
    print(f"\n✅ Cleanup complete! Freed {format_size(freed_space)}")


def clean_all(outputs=True, caches=True, logs=True, force=False):
    """Clean all output directories and cache files"""
    items_to_clean = []
    
    if outputs:
        items_to_clean.extend(get_output_directories())
    
    if caches:
        items_to_clean.extend(get_cache_directories())
    
    if logs:
        # Add log files
        project_root = Path(__file__).parent.parent
        log_files = list(project_root.glob("*.log"))
        log_files.extend(project_root.glob("logs/*"))
        items_to_clean.extend([f for f in log_files if f.exists()])
    
    if not items_to_clean:
        print("✨ Nothing to clean!")
        return
    
    # Calculate total size
    total_size = 0
    for item in items_to_clean:
        if item.is_dir():
            analysis = analyze_directory(item)
            total_size += analysis['size']
        elif item.is_file():
            try:
                total_size += item.stat().st_size
            except (OSError, PermissionError):
                continue
    
    print(f"🗑️  Found {len(items_to_clean)} items to clean")
    print(f"💾 Total space: {format_size(total_size)}")
    
    if not force:
        response = input("Proceed with cleanup? (yes/no): ").strip().lower()
        if response != 'yes':
            print("❌ Cleanup cancelled.")
            return
    
    # Perform cleanup
    freed_space = 0
    for item in items_to_clean:
        try:
            if item.is_dir():
                analysis = analyze_directory(item)
                shutil.rmtree(item)
                freed_space += analysis['size']
            elif item.is_file():
                size = item.stat().st_size
                item.unlink()
                freed_space += size
        except (OSError, PermissionError):
            continue
    
    print(f"✅ Cleanup complete! Freed {format_size(freed_space)}")


def list_outputs():
    """List all output directories and their sizes"""
    output_dirs = get_output_directories()
    cache_dirs = get_cache_directories()
    large_files = get_large_files(min_size_mb=10)
    
    print("📊 SCU Training Output Analysis")
    print("=" * 50)
    
    total_size = 0
    
    if output_dirs:
        print(f"\n📁 Output Directories ({len(output_dirs)}):")
        for directory in output_dirs:
            analysis = analyze_directory(directory)
            rel_path = directory.relative_to(Path.cwd())
            print(f"  {rel_path}: {format_size(analysis['size'])}")
            total_size += analysis['size']
    
    if cache_dirs:
        print(f"\n🗃️  Cache Directories ({len(cache_dirs)}):")
        for directory in cache_dirs:
            analysis = analyze_directory(directory)
            rel_path = directory.relative_to(Path.cwd())
            print(f"  {rel_path}: {format_size(analysis['size'])}")
            total_size += analysis['size']
    
    if large_files:
        print(f"\n📄 Large Files ({len(large_files)}):")
        for file_path in large_files:
            try:
                size = file_path.stat().st_size
                rel_path = file_path.relative_to(Path.cwd())
                print(f"  {rel_path}: {format_size(size)}")
                total_size += size
            except (OSError, PermissionError):
                continue
    
    print(f"\n💾 Total disk usage: {format_size(total_size)}")


def main():
    parser = argparse.ArgumentParser(description='SCU Training Output Cleanup Utility')
    parser.add_argument('--interactive', action='store_true', help='Interactive cleanup mode')
    parser.add_argument('--list', action='store_true', help='List all outputs without cleaning')
    parser.add_argument('--all', action='store_true', help='Clean all outputs, caches, and logs')
    parser.add_argument('--outputs', action='store_true', help='Clean only output directories')
    parser.add_argument('--caches', action='store_true', help='Clean only cache directories')
    parser.add_argument('--logs', action='store_true', help='Clean only log files')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--min-size', type=int, default=50, help='Minimum file size in MB for large file detection')
    
    args = parser.parse_args()
    
    if args.list:
        list_outputs()
    elif args.interactive:
        interactive_cleanup()
    elif args.all:
        clean_all(outputs=True, caches=True, logs=True, force=args.force)
    elif args.outputs or args.caches or args.logs:
        clean_all(outputs=args.outputs, caches=args.caches, logs=args.logs, force=args.force)
    else:
        # Default to interactive mode
        interactive_cleanup()


if __name__ == "__main__":
    main()
