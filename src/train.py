import argparse
import torch
import utils
from oxford_pet import SimpleOxfordPetDataset
import torchvision.transforms.functional as TF
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate,ComboLoss,DiceLoss
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import trange


img_size = 512

device= torch.device("cuda:1" if (torch.cuda.is_available() ) else "cpu")



def trainmodel(model, dataloader, optimizer, criterion):
   
    model.train()
    total_loss = 0
    count = 0
    for data in enumerate(dataloader):
        # print(data)
        images =data[1]["image"]
        target = data[1]["mask"]
        # print(target.shape)
        images=images.to(device)
        target = target.to(device)
        
        target = target.long()
        target = target.squeeze(1)
        #target = torch.argmax(target, dim=1)
       #S print("Target unique values:", torch.unique(target))
        optimizer.zero_grad()
        output = model(images)
        
        if isinstance(output,dict):
            output = output['out']
        loss = criterion(output,target) 

        #loss.requires_grad = True
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count+=1
    return total_loss/count #return avg loss

def train(args):
    # implement the training function here
    #步驟0. 是否使用CUDA
    best_loss = float('inf')
    root=args.data_path
    # mode ="train"
    milestones = [30,50, 100, 150, 200]#represents 到eopch 多少的時候就降低lr
    print(device)
    # 步驟1. data loader處理
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.GaussianBlur(p=0.2),
        # A.GridDropout(ratio=0.2, p=0.5),
        # A.CLAHE(),
        A.RGBShift(),
        # A.RandomCrop(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    
    val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ])
    dataset_train = SimpleOxfordPetDataset(root,"train",train_transform,preprocess=True)
    dataset_val = SimpleOxfordPetDataset(root,"valid",val_transform,preprocess=True)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,num_workers=4, shuffle=True,  pin_memory=True,
    prefetch_factor=2) # 預先讀取資料
    dataloader_val   = torch.utils.data.DataLoader(dataset_val,   batch_size=args.batch_size, num_workers=4,shuffle=False ,pin_memory=True,
    prefetch_factor=2)  # 預先讀取資料
    print(len(dataloader_train))
    print(len(dataloader_val))
    #start Tensorboard Interface
    # Default directory "runs"
    # writer = SummaryWriter() 

    # 步驟2. 模型宣告
    model = ResNet34_UNet(n_channels=3, n_classes=2).to(device)   
     #第55行改第87行就要改
    # Add on Tensorboard the Model Graph
    # img_reference_dummy = torch.randn(1,3,img_size,img_size)
    # img_test_dummy = torch.randn(1,3,img_size,img_size)
    # writer.add_graph(model,[img_reference_dummy,img_test_dummy])
    # 步驟3. loss function宣告
    #criterion = ComboLoss(0.7).to(device)
    criterion = ComboLoss(1).to(device)
    # 步驟4. optimator宣告
    if(args.optimizer=="Adam"):
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=args.learning_rate)
    elif(args.optimizer=="AdamW"):
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4, lr=args.learning_rate)
    elif(args.optimizer=="SGD"):
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4,lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # 步驟5. 模型開始訓練
    # https://github.com/TommyHuang821/Pytorch_DL_Implement/blob/main/10_pytorch_SemanticSegmentation_VOC2007.ipynb
    
    log_loss_train=[]
    log_loss_val=[]
    for epoch in trange(args.epochs, desc="Training Progress", unit="epoch"):
        # train
        train_loss = trainmodel(model, dataloader_train, optimizer, criterion)
        log_loss_train.append([epoch,train_loss])
        scheduler.step()
        # eval
        val_loss = evaluate(model, dataloader_val, criterion)
        log_loss_val.append([epoch,val_loss])
        if val_loss <best_loss :
            best_loss = val_loss
            torch.save(model.state_dict(),'saved_models/ResNet34_UNet0324_2.pth')#第55行改第87行就要改
        
        print('\nlearning rate:{}'.format(scheduler.get_last_lr()[0]))
        print('CNN[epoch: [{}/{}], loss(train):{:.5f}'.format(
            epoch+1, args.epochs, train_loss))
        print('CNN[epoch: [{}/{}], loss(val):{:.5f}'.format(
            epoch+1, args.epochs, val_loss))                    
    print('training done.')
    return log_loss_train, log_loss_val


    # print(args.data_path)
   
   

    # assert False, "train Not implemented yet!"

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--optimizer', '-opt', type=str, default="Adam", help='optimizer')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    log_loss_train ,log_loss_val=train(args)
    utils.draw_loss(log_loss_train,log_loss_val,"train.png")
    
    
