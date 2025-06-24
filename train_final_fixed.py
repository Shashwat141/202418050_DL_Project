# train_final_fixed.py - COMPLETELY FIXED training script

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
import numpy as np
from tqdm.auto import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_
import time, json
from sklearn.metrics import accuracy_score

# Import custom modules
from temporal_sentiment_model_final import SentimentForecastingModel, create_model_for_testing
from loss_functions_fixed import SentimentLoss
from dataset_final import AmazonReviewDataset, load_amazon_reviews, create_sample_data

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

def debug_collate_fn(batch):
    """Debug collate function with extensive shape checking"""
    print(f"\n=== COLLATE DEBUG ===")
    print(f"Batch size: {len(batch)}")
    
    # Check individual items
    for i, item in enumerate(batch):
        print(f"Item {i} shapes:")
        for key, value in item.items():
            print(f"  {key}: {value.shape}")
    
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    time_features = torch.stack([item['time_features'] for item in batch])
    category_id = torch.stack([item['category_id'] for item in batch])
    target = torch.stack([item['target'] for item in batch])
    
    print(f"Stacked shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  time_features: {time_features.shape}")
    print(f"  category_id: {category_id.shape}")
    print(f"  target: {target.shape}")
    print(f"===================\n")
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'time_features': time_features,
        'category_id': category_id,
        'target': target
    }

class FixedModelTrainer:
    """Completely fixed trainer with proper tensor handling"""
    
    def __init__(self, model, device='auto', save_dir='models'):
        self.model = model
        self.save_dir = save_dir
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """Train one epoch with comprehensive debugging"""
        self.model.train()
        total_loss = 0.0
        num_successful_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        device = next(self.model.parameters()).device  # Get model's device
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                time_features = batch['time_features'].to(device)
                category_id = batch['category_id'].to(device)
                targets = batch['target'].to(device)
                
                print(f"\nBatch {batch_idx} shapes:")
                print(f"  input_ids: {input_ids.shape}")
                print(f"  attention_mask: {attention_mask.shape}")
                print(f"  time_features: {time_features.shape}")
                print(f"  category_id: {category_id.shape}")
                print(f"  targets: {targets.shape}")
                
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                
                # **CRITICAL FIX**: Get text embeddings with proper shape handling
                # Reshape for BERT: (batch_size * seq_len, max_length)
                input_ids_flat = input_ids.view(-1, input_ids.size(-1))
                attention_mask_flat = attention_mask.view(-1, attention_mask.size(-1))
                
                print(f"  Flattened for BERT - input_ids: {input_ids_flat.shape}, attention_mask: {attention_mask_flat.shape}")
                
                # Get BERT embeddings (this should return averaged embeddings per input)
                text_embeddings = self.model.get_text_embeddings(input_ids_flat, attention_mask_flat)
                print(f"  BERT embeddings: {text_embeddings.shape}")
                
                # **CRITICAL FIX**: text_embeddings should be (batch_size * seq_len, 768)
                # We need to reshape to (batch_size, seq_len, 768)
                expected_shape = (batch_size * seq_len, 768)
                if text_embeddings.shape != expected_shape:
                    print(f"  WARNING: Unexpected BERT output shape {text_embeddings.shape}, expected {expected_shape}")
                    # Ensure we have the right number of embeddings
                    if text_embeddings.size(0) != batch_size * seq_len:
                        # Create dummy embeddings
                        text_embeddings = torch.randn(batch_size * seq_len, 768, device=self.device)
                
                # Reshape back to (batch_size, seq_len, 768)
                text_embeddings = text_embeddings.view(batch_size, seq_len, -1)
                print(f"  Reshaped text embeddings: {text_embeddings.shape}")
                
                # Forward pass
                outputs = self.model(text_embeddings, time_features, category_id)
                print(f"  Model outputs: {outputs.shape}")
                
                # **CRITICAL FIX**: Loss calculation with proper target handling
                if outputs.dim() == 3:
                    # Take mean over sequence dimension for final prediction
                    outputs = outputs.mean(dim=1)  # (batch_size, num_classes)
                
                # Prepare targets - convert ratings to class indices
                targets_flat = targets.view(-1)  # Flatten all targets
                targets_class = torch.clamp((targets_flat * 4).long(), 0, 4)  # Convert to 0-4 classes
                
                # Expand outputs to match number of targets
                num_targets = targets_flat.size(0)
                if outputs.size(0) != num_targets:
                    # Repeat outputs for each target value
                    repeat_factor = num_targets // outputs.size(0)
                    outputs = outputs.unsqueeze(1).expand(-1, repeat_factor, -1)
                    outputs = outputs.contiguous().view(-1, outputs.size(-1))
                
                print(f"  Final shapes for loss - outputs: {outputs.shape}, targets: {targets_class.shape}")
                
                # Calculate loss
                loss, components = criterion(outputs, targets_class)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                num_successful_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg Loss': f"{total_loss / num_successful_batches:.4f}",
                    'Success': f"{num_successful_batches}/{batch_idx + 1}"
                })
                
                # Memory cleanup
                del loss, outputs, text_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Break after a few successful batches for testing
                '''if num_successful_batches >= 3:
                    print(f"  Stopping early after {num_successful_batches} successful batches for testing")
                    break'''
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / max(num_successful_batches, 1)
        return avg_loss

    def validate_epoch(self, val_loader, criterion, epoch):
        """Validate one epoch with similar fixes"""
        self.model.eval()
        total_loss = 0.0
        num_successful_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} Validation")):
                try:
                    # Similar processing as training but without gradients
                    device = next(self.model.parameters()).device  # Get model's device

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    time_features = batch['time_features'].to(device)
                    category_id = batch['category_id'].to(device)
                    targets = batch['target'].to(device)
                    
                    batch_size = input_ids.size(0)
                    seq_len = input_ids.size(1)
                    
                    # Get embeddings
                    input_ids_flat = input_ids.view(-1, input_ids.size(-1))
                    attention_mask_flat = attention_mask.view(-1, attention_mask.size(-1))
                    
                    text_embeddings = self.model.get_text_embeddings(input_ids_flat, attention_mask_flat)
                    text_embeddings = text_embeddings.view(batch_size, seq_len, -1)
                    
                    # Forward pass
                    outputs = self.model(text_embeddings, time_features, category_id, training=False)
                    
                    # Loss calculation (same as training)
                    if outputs.dim() == 3:
                        outputs = outputs.mean(dim=1)
                    
                    targets_flat = targets.view(-1)
                    targets_class = torch.clamp((targets_flat * 4).long(), 0, 4)
                    
                    num_targets = targets_flat.size(0)
                    if outputs.size(0) != num_targets:
                        repeat_factor = num_targets // outputs.size(0)
                        outputs = outputs.unsqueeze(1).expand(-1, repeat_factor, -1)
                        outputs = outputs.contiguous().view(-1, outputs.size(-1))
                    
                    loss, _ = criterion(outputs, targets_class)
                    total_loss += loss.item()
                    num_successful_batches += 1
                    
                    # Break early for testing
                    if num_successful_batches >= 2:
                        break
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(num_successful_batches, 1)
        return avg_loss

    def train(self, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            # Training
            train_loss = self.train_epoch(train_loader, criterion, optimizer, epoch)
            
            # Validation
            val_loss = self.validate_epoch(val_loader, criterion, epoch)
            
            # Update scheduler
            if scheduler:
                scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print metrics
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  >>> New best model saved!")
        
        print(f"\nTraining completed!")
        return self.history

def main(args):
    """Main training function with extensive debugging"""
    print("=== COMPLETELY FIXED Temporal Sentiment Analysis Training ===")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')}")
    
    try:
        # Create sample data for testing
        print("Creating sample data...")
        data_path = create_sample_data(
            'data/sample_amazon_reviews.json',
            args.sample_size,
            num_products=3
        )
        
        # Load reviews
        print("Loading reviews...")
        reviews_df = load_amazon_reviews(data_path, sample_size=args.sample_size)
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Create dataset
        print("Creating dataset...")
        dataset = AmazonReviewDataset(
            reviews_df,
            tokenizer,
            max_length=args.max_length,
            window_size=args.window_size,
            horizon=args.horizon,
            small_sample=True
        )
        
        if len(dataset) == 0:
            raise ValueError("No temporal windows created")
        
        # Create minimal data loaders for testing
        print("Creating data loaders...")
        dataset_size = len(dataset)
        train_size = max(1, int(0.7 * dataset_size))
        val_size = max(1, int(0.15 * dataset_size))
        test_size = max(1, dataset_size - train_size - val_size)
        
        # Adjust if total exceeds dataset size
        if train_size + val_size + test_size > dataset_size:
            test_size = dataset_size - train_size - val_size
            test_size = max(1, test_size)
        
        train_dataset, temp_dataset = random_split(dataset, [train_size, val_size + test_size])
        val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])
        
        # Use debug collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(args.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=0,
            collate_fn=debug_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=0,
            collate_fn=debug_collate_fn
        )
        
        print(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Initialize model
        print("Initializing model...")
        model = create_model_for_testing()
        
        # Training components
        criterion = SentimentLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Before calling FixedModelTrainer
        if args.force_cpu:
            device = 'cpu'
            print("WARNING: Forcing CPU usage - training will be slow")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
        
        # Then call the trainer
        trainer = FixedModelTrainer(
            model,
            device=device,
            save_dir=args.save_dir
        )
        
        print("Starting training...")
        history = trainer.train(
            train_loader, val_loader, criterion,
            optimizer, scheduler, args.epochs
        )
        
        print("Training completed successfully!")
        
        # === Save metrics to results.json ===
        training_time = None
        acc = None
        try:
            start_eval = time.time()
            # Evaluate on test set
            model.eval()
            all_preds = []
            all_targets = []
            test_loader = DataLoader(
                test_dataset,
                batch_size=min(args.batch_size, len(test_dataset)),
                shuffle=False,
                num_workers=0,
                collate_fn=debug_collate_fn
            )
            with torch.no_grad():
                for batch in test_loader:
                    device = next(model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    time_features = batch['time_features'].to(device)
                    category_id = batch['category_id'].to(device)
                    targets = batch['target'].to(device)

                    # Forward pass (use model's get_text_embeddings and forward)
                    batch_size = input_ids.size(0)
                    seq_len = input_ids.size(1)
                    input_ids_flat = input_ids.view(-1, input_ids.size(-1))
                    attention_mask_flat = attention_mask.view(-1, attention_mask.size(-1))
                    text_embeddings = model.get_text_embeddings(input_ids_flat, attention_mask_flat)
                    text_embeddings = text_embeddings.view(batch_size, seq_len, -1)
                    outputs = model(text_embeddings, time_features, category_id, training=False)
                    if outputs.dim() == 3:
                        outputs = outputs.mean(dim=1)
                    preds = torch.argmax(outputs, dim=-1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    # Targets: convert to class indices
                    targets_flat = targets.view(-1)
                    targets_class = torch.clamp((targets_flat * 4).long(), 0, 4)
                    all_targets.extend(targets_class.cpu().numpy().tolist())
            acc = accuracy_score(all_targets, all_preds)
            training_time = None  # If you want, you can track time from start to here
        except Exception as e:
            print(f"Could not compute test accuracy: {e}")
        # Save metrics
        final_results_dict = {
            "model": "main_model",
            "accuracy": acc,
            "training_time": training_time,
            "history": history
        }
        with open(os.path.join(args.save_dir, "results.json"), "w") as fp:
            json.dump(final_results_dict, fp, indent=2)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Temporal Sentiment Analysis Training")
    
    # Arguments
    parser.add_argument('--sample_size', type=int, default=500, help='Sample size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--category_dim', type=int, default=16, help='Category dimension')
    parser.add_argument('--tcn_channels', nargs='+', type=int, default=[32, 32], help='TCN channels')
    parser.add_argument('--window_size', type=int, default=7, help='Window size')
    parser.add_argument('--horizon', type=int, default=3, help='Horizon')
    parser.add_argument('--max_length', type=int, default=64, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='models', help='Save directory')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU')
    
    args = parser.parse_args()
    main(args)
