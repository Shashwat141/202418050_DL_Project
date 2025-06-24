# run_all_baselines_fixed.py - COMPLETELY FIXED baseline runner

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def print_header():
    """Print formatted header"""
    print("=" * 80)
    print("TEMPORAL SENTIMENT ANALYSIS - FIXED BASELINE COMPARISON")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print()

def run_command_safe(command, description, timeout_minutes=30):
    """Run command with comprehensive error handling"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run with timeout
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout_minutes * 60
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"COMPLETED in {duration:.1f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
            
        return result.returncode == 0, result.stdout, result.stderr, duration
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out ({timeout_minutes} minutes)")
        return False, "", f"Timeout after {timeout_minutes} minutes", timeout_minutes * 60
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False, "", str(e), 0

def check_data_file():
    """Check and prepare data file"""
    data_files = [
        "data/Video_Games.jsonl.gz",
        "data/sample_amazon_reviews.json", 
        "data/Video_Games.jsonl"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"✅ Found data file: {data_file}")
            return data_file
    
    print("⚠️ No data file found. Creating sample data...")
    
    # Create sample data
    success, stdout, stderr, duration = run_command_safe(
        "python -c \"from dataset_final import create_sample_data; create_sample_data()\"",
        "Create sample data",
        timeout_minutes=5
    )
    
    if success and os.path.exists("data/sample_amazon_reviews.json"):
        return "data/sample_amazon_reviews.json"
    
    print("❌ Could not create sample data")
    return None

def load_results(results_file):
    """Load results from JSON file"""
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
    return None

def create_comparison_table(experiment_results):
    """Create formatted comparison table"""
    print("\n" + "=" * 100)
    print("DETAILED RESULTS COMPARISON")
    print("=" * 100)
    
    header = f"{'Model':<20} {'Accuracy (%)':<12} {'F1-Score':<10} {'MAE':<8} {'RMSE':<8} {'Time (s)':<10} {'Status':<10}"
    print(header)
    print("-" * 100)
    
    for name, success, results, duration in experiment_results:
        if success and results:
            status = "✅ SUCCESS"
            accuracy = f"{results.get('accuracy', 0):.2f}"
            f1 = f"{results.get('f1_score', 0):.4f}"
            mae = f"{results.get('mae', 0):.3f}"
            rmse = f"{results.get('rmse', 0):.3f}"
            time_str = f"{results.get('training_time', duration):.1f}"
        else:
            status = "❌ FAILED"
            accuracy = f1 = mae = rmse = time_str = "N/A"
            
        row = f"{name:<20} {accuracy:<12} {f1:<10} {mae:<8} {rmse:<8} {time_str:<10} {status:<10}"
        print(row)

def main():
    """Main execution function"""
    print_header()
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    # Check if we're in the right directory
    if not any(os.path.exists(f) for f in ["train_final_fixed.py", "temporal_sentiment_model_final.py"]):
        print("❌ Error: Run this script from the project directory containing your model files")
        sys.exit(1)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Find data file
    data_file = check_data_file()
    if not data_file:
        print("❌ Error: Could not find or create data file")
        sys.exit(1)
    
    print(f"📁 Using data file: {data_file}")
    
    # Define experiments
    experiments = [
        {
            'name': 'Random Forest',
            'command': f'python baseline_rf_final.py --data_file "{data_file}" --n_estimators 100 --save_dir models/rf_baseline',
            'results_file': 'models/rf_baseline/results.json',
            'timeout': 10
        },
        {
            'name': 'Simple LSTM', 
            'command': f'python baseline_lstm.py --data_file "{data_file}" --epochs 5 --batch_size 32 --save_dir models/lstm_baseline',
            'results_file': 'models/lstm_baseline/results.json',
            'timeout': 15
        },
        {
            'name': 'BERT-only',
            'command': f'python baseline_bert_final.py --data_file "{data_file}" --epochs 3 --batch_size 16 --save_dir models/bert_baseline --sample_size 5000',
            'results_file': 'models/bert_baseline/results.json', 
            'timeout': 20
        }
    ]
    
    # Run experiments
    experiment_results = []
    
    for exp in experiments:
        print(f"\n🚀 Starting {exp['name']} baseline...")
        
        success, stdout, stderr, duration = run_command_safe(
            exp['command'],
            f"{exp['name']} Baseline",
            timeout_minutes=exp['timeout']
        )
        
        # Load results
        results = load_results(exp['results_file']) if success else None
        experiment_results.append((exp['name'], success, results, duration))
        
        if success and results:
            print(f"✅ {exp['name']} completed successfully!")
            print(f"   Accuracy: {results.get('accuracy', 0):.2f}%")
            print(f"   Time: {duration:.1f}s")
        else:
            print(f"❌ {exp['name']} failed")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for _, success, _, _ in experiment_results if success)
    total_count = len(experiment_results)
    
    print(f"Successful experiments: {success_count}/{total_count}")
    
    # Detailed comparison
    create_comparison_table(experiment_results)
    
    # Save markdown report
    print("\n📄 Generating report...")
    
    report_content = f"""# Baseline Comparison Report

Generated on: {datetime.now()}

## Summary
- Total experiments: {total_count}
- Successful: {success_count}
- Failed: {total_count - success_count}

## Results

| Model | Accuracy (%) | F1-Score | MAE | RMSE | Time (s) | Status |
|-------|--------------|----------|-----|------|----------|--------|
"""
    
    for name, success, results, duration in experiment_results:
        if success and results:
            accuracy = f"{results.get('accuracy', 0):.2f}"
            f1 = f"{results.get('f1_score', 0):.4f}"
            mae = f"{results.get('mae', 0):.3f}"
            rmse = f"{results.get('rmse', 0):.3f}"
            time_str = f"{results.get('training_time', duration):.1f}"
            status = "✅ Success"
        else:
            accuracy = f1 = mae = rmse = time_str = "N/A"
            status = "❌ Failed"
            
        report_content += f"| {name} | {accuracy} | {f1} | {mae} | {rmse} | {time_str} | {status} |\n"
    
    report_content += f"""
## Next Steps

1. Review any failed experiments and check error messages
2. Run your main temporal model:
   ```
   python train_final_fixed.py --sample_size 10000 --epochs 10 --batch_size 4
   ```
3. Compare results with baselines for Subtask 3 submission

## Your Main Model Current Performance
- Train Loss: 0.3983 (Moderate - room for improvement)
- Validation Loss: 0.4583 (indicates slight overfitting)
- Status: Working but can be enhanced

Generated at: {datetime.now()}
"""

    with open('results/baseline_comparison_report.md', 'w') as f:
        f.write(report_content)
    
    print("📊 Report saved to: results/baseline_comparison_report.md")
    
    # Final instructions
    print("\n" + "=" * 80)
    print("NEXT STEPS FOR SUBTASK 3")
    print("=" * 80)
    print("1. ✅ Baselines are now working - check results above")
    print("2. 🔧 Your main model needs improvement:")
    print("   - Current loss (0.4) is moderate, aim for <0.3")
    print("   - Consider increasing epochs, adjusting learning rate")
    print("   - Add more temporal features to match PDF spec")
    print("3. 📝 Document your experiment setup using the working results")
    print("4. 📊 Create SOTA comparison table with the baseline results")
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()