import torch
device= torch.device("cuda" if (torch.cuda.is_available() ) else "cpu")
def evaluate(net, dataloader, criterion):
    # implement the evaluation function here
    net.eval()
    lossvalue = 0
    for data in enumerate(dataloader):
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
       
        lossvalue += criterion(output, target)  
    lossvalue /= len(dataloader)
    print("len of data loader:"+str(len(dataloader)))
    return lossvalue

    # assert False, "evaluate Not implemented yet!"