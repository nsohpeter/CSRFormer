"""
Simple script to verify causality analyzer installation

Run this from ~/research/Mask2Former to check everything is set up correctly.
"""

import sys
from pathlib import Path

print("="*80)
print("CAUSALITY ANALYZER - INSTALLATION VERIFICATION")
print("="*80)
print()

# Check current directory
current_dir = Path.cwd()
print(f"Current directory: {current_dir}")

# Check if we're in Mask2Former
if not (current_dir / "mask2former").exists():
    print("❌ ERROR: Not in Mask2Former directory!")
    print("   Please run: cd ~/research/Mask2Former")
    sys.exit(1)
else:
    print("✓ In correct directory")

# Check if causality_analyzer.py exists
analyzer_path = current_dir / "mask2former" / "analysis" / "causality_analyzer.py"
if not analyzer_path.exists():
    print(f"❌ ERROR: causality_analyzer.py not found at:")
    print(f"   {analyzer_path}")
    print("   Please copy: cp ~/Downloads/causality_analyzer.py mask2former/analysis/")
    sys.exit(1)
else:
    print(f"✓ Found causality_analyzer.py")

# Check if scripts exist
run_script = current_dir / "scripts" / "run_causality_analysis.py"
test_script = current_dir / "scripts" / "test_causality_analyzer.py"

if not run_script.exists():
    print(f"❌ ERROR: run_causality_analysis.py not found")
    print("   Please copy: cp ~/Downloads/run_causality_analysis.py scripts/")
    sys.exit(1)
else:
    print("✓ Found run_causality_analysis.py")

if not test_script.exists():
    print(f"❌ ERROR: test_causality_analyzer.py not found")
    print("   Please copy: cp ~/Downloads/test_causality_analyzer.py scripts/")
    sys.exit(1)
else:
    print("✓ Found test_causality_analyzer.py")

# Try to import the module
print()
print("Testing import...")
try:
    sys.path.insert(0, str(current_dir))
    from mask2former.analysis.causality_analyzer import CausalityAnalyzer
    print("✓ Successfully imported CausalityAnalyzer")
except Exception as e:
    print(f"❌ ERROR: Failed to import: {e}")
    sys.exit(1)

# Try to instantiate
try:
    analyzer = CausalityAnalyzer(num_classes=19)
    print("✓ Successfully created analyzer instance")
except Exception as e:
    print(f"❌ ERROR: Failed to create instance: {e}")
    sys.exit(1)

# Check Phase 1 data exists
print()
print("Checking Phase 1 data...")
phase1_dir = current_dir / "experiments" / "stability_analysis" / "results" / "baseline_attention"

if not phase1_dir.exists():
    print(f"⚠️  WARNING: Phase 1 data directory not found:")
    print(f"   {phase1_dir}")
    print("   This is okay if you haven't run Phase 1 extraction yet.")
else:
    print(f"✓ Found Phase 1 data directory")
    
    # Check each domain
    for domain in ["cityscapes", "bdd100k", "mapillary"]:
        domain_dir = phase1_dir / domain
        if domain_dir.exists():
            pkl_files = list(domain_dir.glob("attention_*.pkl"))
            print(f"  ✓ {domain}: {len(pkl_files)} files")
        else:
            print(f"  ⚠️  {domain}: directory not found")

print()
print("="*80)
print("INSTALLATION VERIFIED! ✓")
print("="*80)
print()
print("Ready to run causality analysis!")
print()
print("Next step:")
print("  python scripts/test_causality_analyzer.py")