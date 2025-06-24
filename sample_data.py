# sample_data.py
# Create a small sample dataset for testing

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_sample_data(output_path='data/sample_amazon_reviews.json', num_samples=1000):
    """Create a small sample of Amazon reviews for testing"""
    print(f"Creating sample data with {num_samples} reviews for testing...")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sample data
    sample_data = []
    for i in range(num_samples):
        # Create a random review
        review = {
            'reviewerID': f'user_{i % 100}',
            'asin': f'product_{i % 10}',
            'reviewText': f'This is a sample review {i}. The product is {"good" if i % 5 > 2 else "bad"}.',
            'overall': float(np.random.randint(1, 6)),  # Rating 1-5
            'summary': f'Sample review {i}',
            'unixReviewTime': 1590000000 + i * 86400,  # Starting May 2020, daily reviews
            'reviewTime': '05 20, 2020'
        }
        sample_data.append(review)

    # Save to disk
    with open(output_path, 'w') as f:
        for review in sample_data:
            f.write(json.dumps(review) + '\n')

    print(f"Created sample data with {len(sample_data)} reviews at '{output_path}'")
    return output_path

if __name__ == "__main__":
    # Create sample data
    create_sample_data(num_samples=5000)
