import torch
device= torch.device("cuda:1" if (torch.cuda.is_available() ) else "cpu")
import torch.nn as nn

def evaluate(net, dataloader, criterion):
    # implement the evaluation function here
    net.eval()
    lossvalue = 0
    for data in enumerate(dataloader):
        # print(data)
        images =data[1]["image"]
        target = data[1]["mask"]
        # print("Target unique values:", torch.unique(target))
        
        # print(target.shape)
        images=images.to(device)
        target = target.to(device).long()
         # 確保 target 維度正確
        if target.dim() == 3:
                target = target.unsqueeze(1)
        with torch.no_grad():
            output = net(images)
            if isinstance(output,dict):
                output = output['out']
       
        lossvalue += criterion(output, target)
        
    lossvalue /= len(dataloader)
    # print("len of data loader:"+str(len(dataloader)))
    return lossvalue

    # assert False, "evaluate Not implemented yet!"

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # 確保 pred 維度正確
        pred = torch.sigmoid(pred)
        
        # 如果 target 是 (B, H, W)，轉成 (B, 1, H, W)
        if target.dim() == 3:
            target = target.unsqueeze(1)
           

        # 計算 Dice
        intersection = torch.sum(pred * target, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)

        return 1 - dice.mean()

class ComboLoss(nn.Module):
    def __init__(self, weight=0.5):
        super(ComboLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.weight = weight

    def forward(self, pred, target):
        # 確保 target 是 int 類型 (CrossEntropyLoss需求)
        ce_loss = self.ce(pred, target.squeeze(1).long())

        # Dice Loss 針對多分類，需要轉成 one-hot 格式
        target_one_hot = torch.nn.functional.one_hot(target.squeeze(1), num_classes=pred.shape[1]).permute(0, 3, 1, 2)
        dice_loss = self.dice(pred, target_one_hot.float())

        return self.weight * ce_loss + (1 - self.weight) * dice_loss
