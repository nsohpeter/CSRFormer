#!/usr/bin/env python3
"""
Script to parse Mask2Former evaluation logs and extract mIoU scores
"""

import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

class EvaluationParser:
    def __init__(self, results_dir: Path = Path("experiments/evaluation_results")):
        self.results_dir = results_dir
        self.miou_csv = results_dir / "miou_scores" / "all_models.csv"
        self.raw_logs_dir = results_dir / "raw_logs"
        
    def parse_log(self, log_file: Path) -> Optional[Dict[str, str]]:
        """Parse log file to extract mIoU"""
        print(f"Parsing: {log_file.name}")
        
        if not log_file.exists():
            print(f"Warning: Log file not found: {log_file}")
            return None
            
        # Extract model and dataset from filename
        filename = log_file.stem
        
        if 'baseline' in filename:
            model_name = 'Baseline_M2F_Swin-B'
        elif 'cmformer' in filename and 'swinb' in filename:
            model_name = 'CMFormer_Swin-B'
        elif 'cmformer' in filename and 'tiny' in filename:
            model_name = 'CMFormer_Swin-T'
        else:
            model_name = filename
            
        if 'bdd100k' in filename:
            dataset = 'BDD100K'
        elif 'mapillary' in filename:
            dataset = 'Mapillary'
        elif 'gta5' in filename:
            dataset = 'GTA5'
        elif 'cityscapes' in filename:
            dataset = 'Cityscapes'
        else:
            dataset = 'Unknown'
        
        # Read log file and search for mIoU
        miou = None
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Pattern 1: Detectron2 OrderedDict format
            # Example: OrderedDict([('sem_seg', {'mIoU': 30.14030693010279,
            pattern1 = r"'mIoU':\s*(\d+\.\d+)"
            match1 = re.search(pattern1, content)
            if match1:
                miou = float(match1.group(1))
            
            # Pattern 2: Detectron2 copypaste format (comma-separated values)
            # Example: copypaste: 30.1403,66.4943,47.9962,77.5354
            if miou is None:
                pattern2 = r'copypaste:\s*(\d+\.\d+),\d+\.\d+,\d+\.\d+,\d+\.\d+'
                match2 = re.search(pattern2, content)
                if match2:
                    miou = float(match2.group(1))
            
            # Pattern 3: Direct sem_seg/mIoU format
            if miou is None:
                pattern3 = r'sem_seg/mIoU[:\s]+(\d+\.\d+)'
                match3 = re.search(pattern3, content)
                if match3:
                    miou = float(match3.group(1))
            
            # Pattern 4: Generic IoU pattern (fallback)
            if miou is None:
                pattern4 = r'(?:^|\s)IoU[:\s]+(\d+\.\d+)'
                match4 = re.search(pattern4, content)
                if match4:
                    miou = float(match4.group(1))
        
        if miou is None:
            print(f"Warning: Could not find mIoU in {log_file.name}")
            print("Searched for patterns:")
            print("  - 'mIoU': XX.XX (OrderedDict)")
            print("  - copypaste: XX.XX,... (copypaste format)")
            print("  - sem_seg/mIoU: XX.XX")
            return None
            
        print(f"  ✓ Found: {dataset} mIoU = {miou:.2f}%")
        return {
            'model': model_name,
            'dataset': dataset,
            'miou': f"{miou:.2f}",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    
    def print_summary(self, results: List[Dict[str, str]]):
        """Print formatted summary"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS SUMMARY")
        print("="*70)
        
        for result in results:
            print(f"{result['model']:25s} | {result['dataset']:12s} | {result['miou']:>6s}%")
        
        print("="*70)
    
    def scan_all_logs(self):
        """Scan all log files in raw_logs directory"""
        if not self.raw_logs_dir.exists():
            print(f"Error: {self.raw_logs_dir} does not exist")
            return
        
        log_files = list(self.raw_logs_dir.glob("*.log"))
        if not log_files:
            print(f"No .log files found in {self.raw_logs_dir}")
            return
        
        print(f"Found {len(log_files)} log files to parse\n")
        
        results = []
        for log_file in sorted(log_files):
            result = self.parse_log(log_file)
            if result:
                results.append(result)
            print()  # Blank line between files
        
        if results:
            self.print_summary(results)
            print(f"\n✅ Parsed {len(results)}/{len(log_files)} evaluation logs")
            print(f"💡 Update experiments/evaluation_results/miou_scores/all_models.csv with these values")
        else:
            print("No results extracted from log files")

def main():
    parser = argparse.ArgumentParser(description="Parse Mask2Former evaluation logs")
    parser.add_argument('--log', type=str, help='Path to specific log file')
    parser.add_argument('--scan-all', action='store_true', help='Scan all logs')
    parser.add_argument('--results-dir', type=str, default='experiments/evaluation_results')
    
    args = parser.parse_args()
    
    eval_parser = EvaluationParser(results_dir=Path(args.results_dir))
    
    if args.scan_all:
        eval_parser.scan_all_logs()
    elif args.log:
        result = eval_parser.parse_log(Path(args.log))
        if result:
            eval_parser.print_summary([result])
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
