import argparse
import torch
import utils
from oxford_pet import SimpleOxfordPetDataset
import torchvision.transforms.functional as TF
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter

img_size = 572

device= torch.device("cuda" if (torch.cuda.is_available() ) else "cpu")

def trainmodel(model, dataloader, optimizer, criterion):
   
    model.train()
    for data in enumerate(dataloader):
        # print(data)
        images =data[1]["image"]
        target = data[1]["mask"]
        # print(target.shape)
        images=images.to(device)
        target = target.to(device)
        target = target.long()
        target = torch.argmax(target, dim=1)
        
        optimizer.zero_grad()
        output = model(images)
        if isinstance(output,dict):
            output = output['out']
        loss = criterion(output,target)  
        loss.backward()
        optimizer.step()

def train(args):
    # implement the training function here
    #步驟0. 是否使用CUDA
    best_loss = float('inf')
    root=args.data_path
    # mode ="train"
    milestones = [50, 100, 150, 200]#represents 到eopch 多少的時候就降低lr
    print(device)
    # 步驟1. data loader處理
    dataset_train = SimpleOxfordPetDataset(root,"train")
    dataset_val = SimpleOxfordPetDataset(root,"valid")
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val   = torch.utils.data.DataLoader(dataset_val,   batch_size=args.batch_size, shuffle=False)
    print(len(dataloader_train))
    print(len(dataloader_val))
    #start Tensorboard Interface
    # Default directory "runs"
    # writer = SummaryWriter() 

    # 步驟2. 模型宣告
    model = ResNet34_UNet(n_channels=3, n_classes=1).to(device)   
     #第55行改第87行就要改
    # Add on Tensorboard the Model Graph
    # img_reference_dummy = torch.randn(1,3,img_size,img_size)
    # img_test_dummy = torch.randn(1,3,img_size,img_size)
    # writer.add_graph(model,[img_reference_dummy,img_test_dummy])
    # 步驟3. loss function宣告
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # 步驟4. optimator宣告
    if(args.optimizer=="Adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif(args.optimizer=="AdamW"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif(args.optimizer=="SGD"):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # 步驟5. 模型開始訓練
    # https://github.com/TommyHuang821/Pytorch_DL_Implement/blob/main/10_pytorch_SemanticSegmentation_VOC2007.ipynb
    
    log_loss_train=[]
    log_loss_val=[]
    for epoch in range(args.epochs):
        # train
        trainmodel(model, dataloader_train, optimizer, criterion)
        scheduler.step()
        # eval
        if (epoch % 5 == 0) | (epoch == (args.epochs-1)):
            val_loss = evaluate(model, dataloader_val, criterion)
            log_loss_val.append([epoch,val_loss.detach().cpu().numpy()])
            if val_loss <best_loss :
                best_loss = val_loss
                torch.save(model.state_dict(),'saved_models/ResNet34Unet.pth')#第55行改第87行就要改
            print('\n learning rate:{}'.format(scheduler.get_last_lr()[0]))
            print('CNN[epoch: [{}/{}], loss(val):{:.5f}'.format(
                epoch+1, args.epochs, val_loss))                    
    print('training done.')
    return log_loss_val


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
    log_loss_val=train(args)
    print(log_loss_val)
