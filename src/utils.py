import numpy as np
import torch
import matplotlib.pyplot as plt
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    
    #print(pred_mask)
    #print(gt_mask)
    pred_mask = torch.sigmoid(pred_mask)
    # pred_mask = (pred_mask > 0.5).float()
    #print("Pred Mask After Binarization:", pred_mask.unique())
    #print("GT Mask Unique Values:", gt_mask.unique())
    intersection = torch.sum(pred_mask * gt_mask) * 2.0
    dice = intersection / (torch.sum(pred_mask) + torch.sum(gt_mask))
  
    return dice
    # assert False, "Not implemented dice score yet!"

def draw_loss(train_loss, val_loss, file_name):
   # Extract only the loss values (second value of each [epoch, loss] pair)
    # If train_loss and val_loss are lists of tensors:
    train_losses = [x.item() if isinstance(x, torch.Tensor) else x for x in train_loss]
    val_losses = [x.item() if isinstance(x, torch.Tensor) else x for x in val_loss]

    # Plot the losses
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='orange')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()
