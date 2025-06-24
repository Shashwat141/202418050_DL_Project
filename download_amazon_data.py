#!/usr/bin/env python3
# download_amazon_data.py - Easy Amazon Reviews 2023 Dataset Downloader

import os
import sys
import gzip
import json
import requests
from tqdm import tqdm

# Category information for beginners
CATEGORY_INFO = {
    # Small categories (recommended for testing)
    "All_Beauty": {"size": "701.5K reviews", "difficulty": "BEGINNER", "description": "Beauty products"},
    "Digital_Music": {"size": "130.4K reviews", "difficulty": "BEGINNER", "description": "Digital music products"},
    "Gift_Cards": {"size": "152.4K reviews", "difficulty": "BEGINNER", "description": "Gift cards"},
    "Handmade_Products": {"size": "664.2K reviews", "difficulty": "BEGINNER", "description": "Handmade items"},
    "Health_and_Personal_Care": {"size": "494.1K reviews", "difficulty": "BEGINNER", "description": "Personal care items"},
    "Magazine_Subscriptions": {"size": "71.5K reviews", "difficulty": "BEGINNER", "description": "Magazine subscriptions"},
    "Subscription_Boxes": {"size": "16.2K reviews", "difficulty": "BEGINNER", "description": "Subscription boxes"},

    # Medium categories (good for learning)
    "Amazon_Fashion": {"size": "2.5M reviews", "difficulty": "INTERMEDIATE", "description": "Fashion items"},
    "Appliances": {"size": "2.1M reviews", "difficulty": "INTERMEDIATE", "description": "Home appliances"},
    "Baby_Products": {"size": "6.0M reviews", "difficulty": "INTERMEDIATE", "description": "Baby products"},
    "CDs_and_Vinyl": {"size": "4.8M reviews", "difficulty": "INTERMEDIATE", "description": "Music CDs and vinyl"},
    "Industrial_and_Scientific": {"size": "5.2M reviews", "difficulty": "INTERMEDIATE", "description": "Industrial products"},
    "Musical_Instruments": {"size": "3.0M reviews", "difficulty": "INTERMEDIATE", "description": "Musical instruments"},
    "Software": {"size": "4.9M reviews", "difficulty": "INTERMEDIATE", "description": "Software products"},
    "Video_Games": {"size": "4.6M reviews", "difficulty": "INTERMEDIATE", "description": "Video games"},

    # Large categories (intermediate users)
    "Arts_Crafts_and_Sewing": {"size": "9.0M reviews", "difficulty": "ADVANCED", "description": "Arts and crafts"},
    "Automotive": {"size": "20.0M reviews", "difficulty": "ADVANCED", "description": "Automotive products"},
    "Beauty_and_Personal_Care": {"size": "23.9M reviews", "difficulty": "ADVANCED", "description": "Beauty products"},
    "Cell_Phones_and_Accessories": {"size": "20.8M reviews", "difficulty": "ADVANCED", "description": "Cell phones"},
    "Grocery_and_Gourmet_Food": {"size": "14.3M reviews", "difficulty": "ADVANCED", "description": "Food products"},
    "Health_and_Household": {"size": "25.6M reviews", "difficulty": "ADVANCED", "description": "Health products"},
    "Kindle_Store": {"size": "25.6M reviews", "difficulty": "ADVANCED", "description": "Kindle books"},
    "Movies_and_TV": {"size": "17.3M reviews", "difficulty": "ADVANCED", "description": "Movies and TV shows"},
    "Office_Products": {"size": "12.8M reviews", "difficulty": "ADVANCED", "description": "Office supplies"},
    "Patio_Lawn_and_Garden": {"size": "16.5M reviews", "difficulty": "ADVANCED", "description": "Garden products"},
    "Pet_Supplies": {"size": "16.8M reviews", "difficulty": "ADVANCED", "description": "Pet supplies"},
    "Sports_and_Outdoors": {"size": "19.6M reviews", "difficulty": "ADVANCED", "description": "Sports equipment"},
    "Tools_and_Home_Improvement": {"size": "27.0M reviews", "difficulty": "ADVANCED", "description": "Tools and hardware"},
    "Toys_and_Games": {"size": "16.3M reviews", "difficulty": "ADVANCED", "description": "Toys and games"},

    # Very large categories (advanced users only)
    "Books": {"size": "29.5M reviews", "difficulty": "EXPERT", "description": "Books"},
    "Clothing_Shoes_and_Jewelry": {"size": "66.0M reviews", "difficulty": "EXPERT", "description": "Clothing and jewelry"},
    "Electronics": {"size": "43.9M reviews", "difficulty": "EXPERT", "description": "Electronics"},
    "Home_and_Kitchen": {"size": "67.4M reviews", "difficulty": "EXPERT", "description": "Home and kitchen items"},
}

def print_header():
    """Print header information"""
    print("=" * 70)
    print("Amazon Reviews 2023 Dataset Downloader")
    print("For Temporal Sentiment Analysis Project")
    print("=" * 70)
    print()

def print_categories():
    """Print available categories organized by difficulty"""
    print("Available Categories (organized by difficulty level):")
    print()

    levels = ["BEGINNER", "INTERMEDIATE", "ADVANCED", "EXPERT"]
    level_descriptions = {
        "BEGINNER": "Small datasets - Perfect for testing and learning",
        "INTERMEDIATE": "Medium datasets - Good for real training",
        "ADVANCED": "Large datasets - For experienced users",
        "EXPERT": "Very large datasets - Advanced users only"
    }

    for level in levels:
        print(f"{level} - {level_descriptions[level]}:")
        categories = [cat for cat, info in CATEGORY_INFO.items() if info["difficulty"] == level]
        for i, category in enumerate(categories, 1):
            info = CATEGORY_INFO[category]
            print(f"  {i:2d}. {category:<30} - {info['size']:<15} - {info['description']}")
        print()

def get_user_choice():
    """Get user's category choice"""
    print("Enter the category name exactly as shown above, or:")
    print("  'list' to see categories again")
    print("  'quit' to exit")
    print()

    while True:
        choice = input("Your choice: ").strip()

        if choice.lower() == 'quit':
            print("Goodbye!")
            sys.exit(0)
        elif choice.lower() == 'list':
            print_categories()
            continue
        elif choice in CATEGORY_INFO:
            return choice
        else:
            print(f"Error: '{choice}' is not a valid category.")
            print("Please enter the category name exactly as shown above.")
            print()

def download_file(url, filepath):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with open(filepath, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), 
                           total=total_size//block_size,
                           desc=f"Downloading {os.path.basename(filepath)}",
                           unit='KB'):
                f.write(data)

        return True
    except Exception as e:
        print(f"Error downloading {filepath}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def download_category(category, download_dir="data"):
    """Download a specific category"""
    print(f"Downloading category: {category}")
    print(f"Description: {CATEGORY_INFO[category]['description']}")
    print(f"Size: {CATEGORY_INFO[category]['size']}")
    print(f"Difficulty level: {CATEGORY_INFO[category]['difficulty']}")
    print()

    # Create download directory
    os.makedirs(download_dir, exist_ok=True)

    # Define file paths
    reviews_file = os.path.join(download_dir, f"{category}.jsonl.gz")
    meta_file = os.path.join(download_dir, f"meta_{category}.jsonl.gz")

    # URLs for the files
    reviews_url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{category}.jsonl.gz"
    meta_url = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz"

    # Check if files already exist
    if os.path.exists(reviews_file) and os.path.exists(meta_file):
        print(f"Files already exist for {category}.")
        overwrite = input("Do you want to re-download them? (y/N): ").strip().lower()
        if overwrite not in ['y', 'yes']:
            print("Using existing files.")
            return reviews_file, meta_file

    print("Downloading files...")
    print("This may take a while depending on your internet connection.")
    print()

    # Download reviews file
    print("1/2 Downloading reviews...")
    if download_file(reviews_url, reviews_file):
        print(f"✓ Reviews downloaded to: {reviews_file}")
    else:
        print("✗ Failed to download reviews")
        return None, None

    # Download metadata file
    print("\n2/2 Downloading metadata...")
    if download_file(meta_url, meta_file):
        print(f"✓ Metadata downloaded to: {meta_file}")
    else:
        print("✗ Failed to download metadata")
        return reviews_file, None

    return reviews_file, meta_file

def verify_download(reviews_file, meta_file):
    """Verify the downloaded files"""
    print("\nVerifying downloaded files...")

    # Check reviews file
    if reviews_file and os.path.exists(reviews_file):
        try:
            with gzip.open(reviews_file, 'rt') as f:
                # Read first few lines to verify format
                for i, line in enumerate(f):
                    if i >= 3:  # Check first 3 lines
                        break
                    json.loads(line)  # This will raise an exception if invalid JSON
            print(f"✓ Reviews file is valid: {reviews_file}")

            # Get file size
            size_mb = os.path.getsize(reviews_file) / (1024 * 1024)
            print(f"  File size: {size_mb:.1f} MB")

        except Exception as e:
            print(f"✗ Reviews file appears corrupted: {e}")
            return False
    else:
        print("✗ Reviews file not found")
        return False

    # Check metadata file
    if meta_file and os.path.exists(meta_file):
        try:
            with gzip.open(meta_file, 'rt') as f:
                # Read first few lines to verify format
                for i, line in enumerate(f):
                    if i >= 3:  # Check first 3 lines
                        break
                    json.loads(line)  # This will raise an exception if invalid JSON
            print(f"✓ Metadata file is valid: {meta_file}")

            # Get file size
            size_mb = os.path.getsize(meta_file) / (1024 * 1024)
            print(f"  File size: {size_mb:.1f} MB")

        except Exception as e:
            print(f"✗ Metadata file appears corrupted: {e}")
    else:
        print("⚠ Metadata file not found (optional)")

    return True

def print_next_steps(category, reviews_file):
    """Print instructions for next steps"""
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE! What to do next:")
    print("=" * 70)
    print()

    print("1. QUICK TEST - Run a small test to verify everything works:")
    print(f"   python train_final_fixed.py --test --category {category}")
    print()

    print("2. SMALL TRAINING - Train on a subset of the data:")
    print(f"   python train_final_fixed.py --data_path {reviews_file} --small_sample --sample_size 10000")
    print()

    print("3. FULL TRAINING - Train on the complete dataset:")
    print(f"   python train_final_fixed.py --data_path {reviews_file}")
    print()

    print("IMPORTANT NOTES:")
    print("• Always start with the quick test first!")
    print("• If you're a beginner, use small training before full training")
    print("• Make sure you have all required Python packages installed:")
    print("  pip install -r requirements.txt")
    print()

    difficulty = CATEGORY_INFO[category]["difficulty"]
    if difficulty in ["ADVANCED", "EXPERT"]:
        print("⚠ WARNING: This is a large dataset!")
        print("• Make sure you have enough disk space and RAM")
        print("• Consider starting with a smaller category first")
        print("• Training may take several hours")

    print("\nFor help, check the README or run: python train_final_fixed.py --help")

def main():
    """Main function"""
    print_header()

    # Check if running from correct directory
    if not os.path.exists('train_final_fixed.py'):
        print("ERROR: This script should be run from the project directory.")
        print("Make sure you're in the same directory as train_final_fixed.py")
        sys.exit(1)

    print_categories()

    # Get user choice
    category = get_user_choice()

    # Confirm choice
    print(f"\nYou selected: {category}")
    print(f"Description: {CATEGORY_INFO[category]['description']}")
    print(f"Size: {CATEGORY_INFO[category]['size']}")
    print(f"Difficulty: {CATEGORY_INFO[category]['difficulty']}")
    print()

    confirm = input("Continue with download? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("Download cancelled.")
        sys.exit(0)

    # Download the category
    reviews_file, meta_file = download_category(category)

    if reviews_file:
        # Verify the download
        if verify_download(reviews_file, meta_file):
            print_next_steps(category, reviews_file)
        else:
            print("\nDownload verification failed. Please try again.")
            sys.exit(1)
    else:
        print("\nDownload failed. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
