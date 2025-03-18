import argparse
import torch
device= torch.device("cuda" if (torch.cuda.is_available() ) else "cpu")
from utils import dice_score
from oxford_pet import SimpleOxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

def test_evaluate(args):
    net = ResNet34_UNet(n_channels=3, n_classes=1).to(device)
    net.load_state_dict(torch.load(args.model))
   
    net.eval()
    lossvalue = 0
    dataset_test = SimpleOxfordPetDataset(args.data_path,"test")
    dataloader_test   = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    print(len(dataloader_test))
    for data in enumerate(dataloader_test):
        # print(data)
        images =data[1]["image"]
        target = data[1]["mask"]
        # print("Target unique values:", torch.unique(target))
        target = target.long()  # 轉成整數類別索引
        # print(target.shape)
        images=images.to(device)
        target = target.to(device)
        target = torch.argmax(target, dim=1)
        with torch.no_grad():
            output = net(images)
            if isinstance(output,dict):
                output = output['out']
       
        lossvalue += dice_score(output, target)  
    lossvalue /= len(dataloader_test)
    print("len of data loader:"+str(len(dataloader_test)))
    return lossvalue




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    dice = test_evaluate(args)
    print("avg_dice_score: ")
    print(dice)

    # assert False, "Not implemented yet!"
