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
    if(args.model_type=='UNet'):
        net = UNet(n_channels=3, n_classes=2).to(device)
    else:
        net = ResNet34_UNet(n_channels=3, n_classes=2).to(device)

    net.load_state_dict(torch.load(args.model))
    
    net.eval()
    lossvalue = 0
    val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ])
    dataset_test = SimpleOxfordPetDataset(args.data_path,"test",val_transform,preprocess=args.data_preprocess)
    dataloader_test   = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,num_workers=4, shuffle=False)
    print("dataloader_test")
    print(len(dataloader_test))
    
    images_list = []
    masks_list = []
    predictions_list = []
    i=0
    for  data in enumerate(dataloader_test):
        # print(data)
        images =data[1]["image"]
        target = data[1]["mask"]
        target = target.long()  # ????????????????????????
        # print(target.shape)
        images=images.to(device)
        target = target.to(device).long()
        if target.dim() == 3:
            target = target.unsqueeze(1)
        # target = torch.argmax(target, dim=1)
        
        with torch.no_grad():
            output = net(images)
            if isinstance(output,dict):
                output = output['out']
            
        lossvalue += dice_score(output, target)  
        
        if i < 4:
            for img_idx in range(4):
              images_list.append(images[img_idx].cpu().permute(1, 2, 0).numpy())
              masks_list.append(target[img_idx].cpu().squeeze().numpy())
              predictions_list.append(torch.argmax(output[img_idx], dim=0).cpu().squeeze().numpy())
        i+=1
            
    lossvalue /= len(dataloader_test)
    print("len of data loader:"+str(len(dataloader_test)))
    
        # Visualization
    fig, axes = plt.subplots(4, 3, figsize=(10, 15))
    for idx in range(4):
        axes[idx, 0].imshow((images_list[idx]))
        axes[idx, 0].set_title("Original Image")
        axes[idx, 1].imshow(masks_list[idx], cmap='gray')
        axes[idx, 1].set_title("Ground Truth Mask")
        axes[idx, 2].imshow(predictions_list[idx], cmap='gray')
        axes[idx, 2].set_title("Predicted Mask")

        for ax in axes[idx]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(args.save_fig_filename)
    # plt.show()
    
    
    
    
    return lossvalue




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--model_type',  type=str, default='UNet', help='UNet/ResNet34UNet')
    parser.add_argument('--data_preprocess',  type=bool, default=False, help='data preprocessing')
    parser.add_argument('--save_fig_filename',  type=str, default='segmentation_visualization.png', help='saving visualize fig')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    dice = test_evaluate(args)
    #  ?????? NumPy
    numpy_dice = dice.detach().cpu().numpy()
    print("avg_dice_score: ")
    print(numpy_dice)

    # assert False, "Not implemented yet!"
