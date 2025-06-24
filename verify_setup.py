# verify_setup.py - Comprehensive setup verification script

import sys
import os
import subprocess
import importlib
import warnings
warnings.filterwarnings('ignore')

def check_python_version():
    """Check Python version compatibility"""
    print("=== Python Version Check ===")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version compatible")
        return True

def check_packages():
    """Check if required packages are installed"""
    print("\n=== Package Installation Check ===")
    
    required_packages = [
        ('torch', '2.0.0'),
        ('transformers', '4.30.0'),
        ('numpy', '1.24.0'),
        ('pandas', '2.0.0'),
        ('sklearn', '1.2.0'),
        ('matplotlib', '3.7.0'),
        ('tqdm', '4.65.0'),
        ('requests', '2.31.0')
    ]
    
    missing_packages = []
    
    for package, min_version in required_packages:
        try:
            # Handle special cases
            if package == 'sklearn':
                import sklearn
                module = sklearn
                package_name = 'scikit-learn'
            else:
                module = importlib.import_module(package)
                package_name = package
            
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
            
        except ImportError:
            print(f"❌ {package_name}: Not installed")
            missing_packages.append(package_name if package != 'sklearn' else 'scikit-learn')
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n=== CUDA Check ===")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def check_file_structure():
    """Check if all required files are present"""
    print("\n=== File Structure Check ===")
    
    required_files = [
        'dataset_fixed.py',
        'temporal_sentiment_model_fixed.py', 
        'loss_functions_fixed.py',
        'train_fixed.py',
        'requirements_fixed.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_imports():
    """Check if custom modules can be imported"""
    print("\n=== Module Import Check ===")
    
    modules_to_test = [
        'dataset_fixed',
        'temporal_sentiment_model_fixed',
        'loss_functions_fixed'
    ]
    
    import_errors = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            import_errors.append(module)
    
    if import_errors:
        print(f"\nImport errors in: {', '.join(import_errors)}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\n=== Basic Functionality Test ===")
    
    try:
        # Test dataset creation
        from dataset_fixed import create_sample_data, load_amazon_reviews, AmazonReviewDataset
        from transformers import BertTokenizer
        
        print("Creating sample data...")
        data_path = create_sample_data('test_data.json', 100, 2)
        
        print("Loading sample data...")
        df = load_amazon_reviews(data_path)
        
        print("Initializing tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        print("Creating dataset...")
        dataset = AmazonReviewDataset(df, tokenizer, window_size=5, horizon=2, 
                                    small_sample=True, force_windows=True)
        
        if len(dataset) > 0:
            print(f"✅ Dataset created with {len(dataset)} windows")
        else:
            print("❌ Dataset empty")
            return False
            
        # Test model creation
        from temporal_sentiment_model_fixed import create_model_for_testing
        
        print("Creating test model...")
        model = create_model_for_testing()
        print(f"✅ Model created with {model.count_parameters():,} parameters")
        
        # Test data item
        print("Testing data loading...")
        sample = dataset[0]
        print(f"✅ Sample loaded - Input shape: {sample['input_ids'].shape}")
        
        # Cleanup
        if os.path.exists('test_data.json'):
            os.remove('test_data.json')
        
        print("✅ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_memory_requirements():
    """Check system memory"""
    print("\n=== Memory Requirements Check ===")
    
    try:
        import psutil
        
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        print(f"Total memory: {total_memory:.1f} GB")
        print(f"Available memory: {available_memory:.1f} GB")
        
        if available_memory < 4:
            print("⚠️  Warning: Less than 4GB available memory")
            print("   Consider using --force_cpu and smaller batch sizes")
        else:
            print("✅ Sufficient memory available")
            
        return True
        
    except ImportError:
        print("⚠️  psutil not available - cannot check memory")
        return True
    except Exception as e:
        print(f"❌ Error checking memory: {e}")
        return False

def run_quick_test():
    """Run a quick end-to-end test"""
    print("\n=== Quick End-to-End Test ===")
    
    try:
        cmd = [
            sys.executable, 'train_fixed.py',
            '--test',
            '--create_sample',
            '--sample_size', '100',
            '--batch_size', '2',
            '--epochs', '1',
            '--force_cpu',
            '--window_size', '3',
            '--horizon', '1'
        ]
        
        print("Running quick training test...")
        print("Command:", ' '.join(cmd))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Quick test completed successfully!")
            return True
        else:
            print("❌ Quick test failed")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Quick test timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running quick test: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 Temporal Sentiment Analysis - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Installation", check_packages),
        ("CUDA Availability", check_cuda),
        ("File Structure", check_file_structure),
        ("Module Imports", check_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Memory Requirements", check_memory_requirements),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready for training.")
        
        # Ask if user wants to run quick test
        try:
            response = input("\nRun quick end-to-end test? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                run_quick_test()
        except KeyboardInterrupt:
            print("\nSkipping quick test.")
            
    else:
        print(f"\n⚠️  {total - passed} checks failed. Please fix issues before training.")
        
        # Provide specific guidance
        print("\n📋 NEXT STEPS:")
        for check_name, result in results:
            if not result:
                if check_name == "Package Installation":
                    print("• Install missing packages: pip install -r requirements_fixed.txt")
                elif check_name == "File Structure":
                    print("• Ensure all Python files are in the same directory")
                elif check_name == "Module Imports":
                    print("• Fix syntax errors in Python files")
                elif check_name == "CUDA Availability":
                    print("• Use --force_cpu flag for CPU-only training")

if __name__ == "__main__":
    main()