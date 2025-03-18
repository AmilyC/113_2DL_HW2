import numpy as np
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    intersection = np.sum(pred_mask[gt_mask==1]) * 2.0
    dice = intersection / (np.sum(pred_mask) + np.sum(gt_mask))
    return dice
    # assert False, "Not implemented dice score yet!"

