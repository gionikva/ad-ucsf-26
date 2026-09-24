import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.models import LightMedSeg


class LightMedSegLoss(nn.Module):
    def __init__(self, ce_only=False, num_classes=2):
        super().__init__()
        
        self.ce_only = ce_only
        
        self.num_classes = num_classes
        
    def forward(self, logits, targets):
        B, C, D, H, W = logits.shape
        
        # Small value to avoid division by zero
        eps = 1e-7
        
        # Per-class weights calculation:
        # ______________________________
        # Total number of voxels across batches
        omega = B * D * H * W
        # Number of target voxels per class 
        counts = targets.sum(dim=(0, 2, 3, 4))
        # weights per class (biased by number of relevant voxels per class)
        w_c = omega / (self.num_classes * counts + eps)
        # normalize weights
        w_c = w_c / w_c.sum() * self.num_classes
        
        # Cross entropy loss (weighted)
        loss_ce = F.cross_entropy(logits, targets, weight=w_c)
        
        # DICE loss calculation:
        # ______________________________
        # Calculate class probabilities per voxel with softmax
        probs = F.softmax(logits, dim=1)
        # Calculate the intersection between predicted masks and target masks per class
        intersection = torch.sum(probs * targets, dim=(2, 3, 4))
        # Calculate the total area of predicted and target masks per class
        probs_sum = probs.sum(dim=[2, 3, 4])
        targets_sum = targets.sum(dim=[2, 3, 4])
        total = probs_sum + targets_sum
        # Calculate the DICE-score per batch and channel
        dice_score = (2. * intersection + eps) / (total + eps)
        dice_loss_standard = 1.0 - dice_score
        
        # Empty mask penalty calculation
        # ______________________________
        # Calculate sum per batch and channel of probabilities and mask
        
        # Empty mask penalty 
        volume_size = probs.shape[2] * probs.shape[3] * probs.shape[4]
        empty_mask_penalty = probs_sum / volume_size
        # Apply if for particular channel and batch the mask sum is equal to zero
        dice_loss = torch.where(targets_sum == 0, empty_mask_penalty, dice_loss_standard)
        
        
        dice_loss_per_class = dice_loss.mean(dim=0)
                        # Apply per-class weights to calculate final dice loss
        loss_dice = torch.sum(dice_loss_per_class * w_c) / self.num_classes        
                
        # Subtract cross-batch mean from 1.0 for dice loss per class
      
        
        # Boundary loss:
        # ______________________________
        # Boundary mask
        dilated = F.max_pool3d(targets, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool3d(-targets, kernel_size=3, stride=1, padding=1)
        M = dilated - eroded
        # Independently calculates the cross-entropy for each class
        bce_voxelwise = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        # Masks out the boundary voxels
        masked_bce = bce_voxelwise * M
        # Total boundary voxels per batch & class
        M_sum = torch.sum(M, dim=(2, 3, 4)) + eps
        # Normalized cross entropy per batch and class
        bdry_loss_per_item_class = torch.sum(masked_bce, dim=(2, 3, 4)) / M_sum
        # Normalized cross entropy per class
        bdry_loss_per_class = bdry_loss_per_item_class.mean(dim=0)
        # Boundary loss (weighted)
        loss_bdry = torch.sum(bdry_loss_per_class * w_c) / self.num_classes
        
        # Total loss calculation
        total_loss = loss_dice + loss_ce + 0.5 * loss_bdry
        if not self.ce_only:
            return total_loss, loss_dice, loss_ce, loss_bdry
        else:
            return loss_ce, torch.tensor(0.), loss_ce, torch.tensor(0.) 
  