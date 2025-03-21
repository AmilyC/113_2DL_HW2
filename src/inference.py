import argparse
import torch
device= torch.device("cuda" if (torch.cuda.is_available() ) else "cpu")
from utils import dice_score
from oxford_pet import SimpleOxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
def test_evaluate(args):
    net = UNet(n_channels=3, n_classes=2).to(device)
    net.load_state_dict(torch.load(args.model))
    
    net.eval()
    lossvalue = 0
    val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ])
    dataset_test = SimpleOxfordPetDataset(args.data_path,"test",val_transform)
    dataloader_test   = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,num_workers=4, shuffle=False)
    print(len(dataloader_test))
    for data in enumerate(dataloader_test):
        # print(data)
        images =data[1]["image"]
        target = data[1]["mask"]
        target = target.long()  # 轉成整數類別索引
        # print(target.shape)
        images=images.to(device)
        target = target.to(device)
       
        # target = torch.argmax(target, dim=1)
        
        with torch.no_grad():
            output = net(images)
            if isinstance(output,dict):
                output = output['out']
                preds = torch.argmax(output, dim=1)
           
            # 顯示前幾個樣本
    num_samples = min(4, len(images))
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        # 顯示原圖
        axs[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis('off')

        # 顯示真實標註
        axs[i, 1].imshow(target[i].cpu().numpy(), cmap='gray')
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis('off')

        # 顯示模型預測結果
        axs[i, 2].imshow(preds[i].cpu().numpy(), cmap='gray')
        axs[i, 2].set_title("Model Prediction")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
        # print("output unique values:", torch.unique(output))
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
    #  轉成 NumPy
    numpy_dice = dice.detach().cpu().numpy()
    print("avg_dice_score: ")
    print(numpy_dice)

    # assert False, "Not implemented yet!"
