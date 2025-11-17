"""
Training Script for Binary Classification with LNR

This script demonstrates the complete two-stage training pipeline:
  Stage 1: Pre-train a model on imbalanced binary data
  Stage 2: Apply LNR and fine-tune

Example usage:
  Stage 1: python train_binary_lnr.py --stage 1 --imbalance 0.1 --epochs 100
  Stage 2: python train_binary_lnr.py --stage 2 --resume checkpoint.pth --threshold 3.0
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

from binary_lnr import BinaryLNR, apply_label_flips, visualize_noise_model
from models import resnet_cifar


def parse_args():
    parser = argparse.ArgumentParser(description='Binary LNR Training')
    
    # General
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                        help='Training stage: 1 (pretrain) or 2 (LNR)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--save-dir', type=str, default='./saved_binary',
                        help='Directory to save models')
    
    # Binary classification settings
    parser.add_argument('--class0', type=int, default=0,
                        help='First class for binary classification')
    parser.add_argument('--class1', type=int, default=1,
                        help='Second class for binary classification')
    parser.add_argument('--imbalance', type=float, default=0.1,
                        help='Imbalance ratio (class1/class0)')
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet32_fe',
                        help='Backbone architecture')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint for stage 2')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    
    # LNR specific (Stage 2)
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='LNR threshold (t_flip)')
    parser.add_argument('--n-passes', type=int, default=2,
                        help='Number of passes to average predictions')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--print-freq', type=int, default=50,
                        help='Print frequency')
    
    return parser.parse_args()


class BinaryImbalancedDataset(Dataset):
    """
    Creates binary imbalanced dataset from CIFAR
    """
    def __init__(self, root, train=True, class0=0, class1=1, 
                 imbalance_ratio=0.1, transform=None):
        """
        Args:
            root: Dataset root
            train: Train or test split
            class0: First class (majority)
            class1: Second class (minority)
            imbalance_ratio: Ratio of class1/class0 samples
            transform: Transforms to apply
        """
        # Load CIFAR
        cifar = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=None
        )
        
        # Get indices for both classes
        targets = np.array(cifar.targets)
        idx0 = np.where(targets == class0)[0]
        idx1 = np.where(targets == class1)[0]
        
        # Apply imbalance to training set only
        if train:
            # Keep all class0 (majority)
            n_class0 = len(idx0)
            
            # Subsample class1 (minority)
            n_class1 = int(n_class0 * imbalance_ratio)
            np.random.seed(42)
            idx1 = np.random.choice(idx1, size=n_class1, replace=False)
        
        # Combine indices
        indices = np.concatenate([idx0, idx1])
        
        # Create subset
        self.cifar = cifar
        self.indices = indices
        self.transform = transform
        
        # Map to binary labels: class0 -> 0, class1 -> 1
        self.class0 = class0
        self.class1 = class1
        
        # Print statistics
        n0 = len(idx0)
        n1 = len(idx1)
        split = "train" if train else "test"
        print(f"  {split.capitalize()} set:")
        print(f"    Class 0 ({class0}): {n0} samples")
        print(f"    Class 1 ({class1}): {n1} samples")
        print(f"    Imbalance ratio: {n1/n0:.3f}")
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get original sample
        real_idx = self.indices[idx]
        img, target = self.cifar[real_idx]
        
        # Convert to binary label
        binary_target = 0 if target == self.class0 else 1
        
        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        
        return idx, img, binary_target


def create_binary_dataloaders(args):
    """Create binary classification dataloaders"""
    
    print(f"\nCreating binary dataset: class {args.class0} vs {args.class1}")
    print(f"Imbalance ratio: {args.imbalance}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Datasets
    train_dataset = BinaryImbalancedDataset(
        root=args.data_path,
        train=True,
        class0=args.class0,
        class1=args.class1,
        imbalance_ratio=args.imbalance,
        transform=train_transform
    )
    
    test_dataset = BinaryImbalancedDataset(
        root=args.data_path,
        train=False,
        class0=args.class0,
        class1=args.class1,
        imbalance_ratio=1.0,  # Balanced test set
        transform=test_transform
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset


def train_epoch(model, classifier, train_loader, criterion, optimizer, 
                epoch, args, noise_info=None):
    """Train for one epoch"""
    
    model.eval()  # Feature extractor in eval mode
    classifier.train()
    
    losses = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (indices, images, targets) in enumerate(train_loader):
        # Move to GPU
        images = images.cuda(args.gpu)
        targets = targets.cuda(args.gpu)
        
        # Apply label flips if in stage 2
        if noise_info is not None:
            targets = apply_label_flips(targets, indices, noise_info)
        
        # Forward pass
        with torch.no_grad():
            features = model(images)
        
        logits = classifier(features.detach())
        loss = criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        losses.append(loss.item())
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if batch_idx % args.print_freq == 0:
            print(f'  Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_time = time.time() - start_time
    avg_loss = np.mean(losses)
    accuracy = 100. * correct / total
    
    print(f'\nEpoch {epoch} Summary:')
    print(f'  Time: {epoch_time:.2f}s')
    print(f'  Loss: {avg_loss:.4f}')
    print(f'  Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def evaluate(model, classifier, test_loader, args):
    """Evaluate model"""
    
    model.eval()
    classifier.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    correct = 0
    total = 0
    
    # Per-class statistics
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for indices, images, targets in test_loader:
            images = images.cuda(args.gpu)
            targets = targets.cuda(args.gpu)
            
            # Forward pass
            features = model(images)
            logits = classifier(features)
            loss = criterion(logits, targets)
            
            # Statistics
            losses.append(loss.item())
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(2):
                mask = targets == i
                class_correct[i] += predicted[mask].eq(targets[mask]).sum().item()
                class_total[i] += mask.sum().item()
    
    avg_loss = np.mean(losses)
    accuracy = 100. * correct / total
    
    print(f'\nTest Results:')
    print(f'  Loss: {avg_loss:.4f}')
    print(f'  Overall Accuracy: {accuracy:.2f}%')
    print(f'  Class 0 Accuracy: {100.*class_correct[0]/class_total[0]:.2f}%')
    print(f'  Class 1 Accuracy: {100.*class_correct[1]/class_total[1]:.2f}%')
    print(f'  Balanced Accuracy: {50.*(class_correct[0]/class_total[0] + class_correct[1]/class_total[1]):.2f}%')
    
    return avg_loss, accuracy, class_correct, class_total


def adjust_learning_rate(optimizer, epoch, args):
    """Cosine learning rate schedule"""
    lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 80)
    print("Binary Classification with Label Noise Rebalancing (LNR)")
    print("=" * 80)
    print(f"\nStage: {args.stage}")
    print(f"Dataset: {args.dataset}")
    print(f"Binary Classes: {args.class0} vs {args.class1}")
    print(f"Imbalance Ratio: {args.imbalance}")
    if args.stage == 2:
        print(f"LNR Threshold: {args.threshold}")
    print()
    
    # Create dataloaders
    train_loader, test_loader, train_dataset = create_binary_dataloaders(args)
    
    # Create model
    print(f"\nCreating model: {args.backbone}")
    model = getattr(resnet_cifar, args.backbone)()
    classifier = getattr(resnet_cifar, 'Classifier')(feat_in=64, num_classes=2)
    
    model = model.cuda(args.gpu)
    classifier = classifier.cuda(args.gpu)
    
    # Load checkpoint if stage 2
    if args.stage == 2:
        if not args.resume:
            raise ValueError("Stage 2 requires --resume checkpoint")
        
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.gpu}')
        model.load_state_dict(checkpoint['model'])
        classifier.load_state_dict(checkpoint['classifier'])
        print("Checkpoint loaded successfully")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.stage == 1:
        # Stage 1: Train both model and classifier
        optimizer = optim.SGD(
            list(model.parameters()) + list(classifier.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        # Stage 2: Only train classifier
        optimizer = optim.SGD(
            classifier.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    
    # Stage 2: Generate noise model
    noise_info = None
    if args.stage == 2:
        print("\n" + "=" * 80)
        print("Generating LNR Noise Model")
        print("=" * 80)
        
        lnr = BinaryLNR(
            model=model,
            classifier=classifier,
            threshold=args.threshold,
            n_passes=args.n_passes,
            device=f'cuda:{args.gpu}'
        )
        
        noise_path = os.path.join(args.save_dir, 'noise_model.pkl')
        noise_info = lnr.generate_noise_model(train_loader, save_path=noise_path)
        
        # Visualize
        predictions = lnr.collect_predictions(train_loader)
        viz_path = os.path.join(args.save_dir, 'noise_model_viz.png')
        try:
            visualize_noise_model(predictions, noise_info, save_path=viz_path)
        except:
            print("Could not create visualization (matplotlib may not be available)")
    
    # Training loop
    print("\n" + "=" * 80)
    print(f"Training Stage {args.stage}")
    print("=" * 80)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f"Learning rate: {lr:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, classifier, train_loader, criterion, optimizer,
            epoch, args, noise_info=noise_info
        )
        
        # Evaluate
        test_loss, test_acc, class_correct, class_total = evaluate(
            model, classifier, test_loader, args
        )
        
        # Save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'args': args
        }
        
        # Save current
        save_path = os.path.join(args.save_dir, f'stage{args.stage}_current.pth')
        torch.save(checkpoint, save_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(args.save_dir, f'stage{args.stage}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"\n*** New best accuracy: {best_acc:.2f}% - Saved to {best_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
