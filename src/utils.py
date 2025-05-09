import numpy as np
import torch
import matplotlib.pyplot as plt
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    eps=1e5
    #print(pred_mask)
    #print(gt_mask)
    pred_mask = torch.sigmoid(pred_mask)
    
    pred_mask = (pred_mask > 0.5).long()
    if pred_mask.shape[1] > 1:  # ????????? one-hot
        pred_mask = torch.argmax(pred_mask, dim=1, keepdim=True)
    
    #print("Pred Mask After Binarization:", pred_mask.unique())
    #print("GT Mask Unique Values:", gt_mask.unique())
    intersection = torch.sum(pred_mask * gt_mask, dim=(1, 2, 3))
    union = torch.sum(pred_mask, dim=(1, 2, 3)) + torch.sum(gt_mask, dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
  #print(dice)
  #print(intersection)
  #print(union)
    return dice.mean()
    # assert False, "Not implemented dice score yet!"

def draw_loss(train_loss, val_loss, file_name):
   # Extract only the loss values (second value of each [epoch, loss] pair)
    train_losses = [x[1].detach().cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in train_loss]
    val_losses = [x[1].detach().cpu().numpy() if isinstance(x[1], torch.Tensor) else x[1] for x in val_loss]
   
    # Plot the losses
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='orange')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()
