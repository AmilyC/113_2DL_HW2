import os
import subprocess
os.system('python src/downloaddata0312.py')
os.system('python src/image_preprocess.py')
UNet_No_pre_cmd = [
    "python", "src/train.py",
    "--data_path", "dataset/oxford-iiit-pet",
    "--epoch", '50',
    "--batch_size", '8',
    "--learning_rate", '0.001',
    "--optimizer", "Adam",
    "--model_type", "UNet",
    '--data_preprocess','False',
    '--save_fig_filename','trainvalUNetNoPre.png',
    '--save_pth_filename','UNetNoPre.pth'

]

UNet_pre_cmd = [
    "python", "src/train.py",
    "--data_path", "dataset/oxford-iiit-pet",
    "--epoch", '50',
    "--batch_size", '8',
    "--learning_rate", '0.001',
    "--optimizer", "Adam",
    "--model_type", "UNet",
    '--data_preprocess','True',
    '--save_fig_filename','trainvalUNetPre.png',
    '--save_pth_filename','UNetPre.pth'

]

ResNet34UNet_No_pre_cmd = [
    "python", "src/train.py",
    "--data_path", "dataset/oxford-iiit-pet",
    "--epoch", '50',
    "--batch_size", '8',
    "--learning_rate", '0.001',
    "--optimizer", "Adam",
    "--model_type", "ResNet34UNet",
    '--data_preprocess','False',
    '--save_fig_filename','trainvalResNet34UNetNoPre.png',
    '--save_pth_filename','ResNet34UNetNoPre.pth'

]

ResNet34UNet_pre_cmd = [
    "python", "src/train.py",
    "--data_path", "dataset/oxford-iiit-pet",
    "--epoch", '50',
    "--batch_size", '8',
    "--learning_rate", '0.001',
    "--optimizer", "Adam",
    "--model_type", "ResNet34UNet",
    '--data_preprocess','True',
    '--save_fig_filename','trainvalResNet34UNetPre.png',
    '--save_pth_filename','ResNet34UNetPre.pth'

]


UNet_No_pre_test=[
    "python", "src/inference.py",
    "--data_path", "dataset/oxford-iiit-pet",
    "--model","saved_models/UNetNoPre.pth",
    "--batch_size",'16',
    "--model_type", "UNet",
    '--data_preprocess','False',
    '--save_fig_filename', 'TestUNetNoPre.png'

]


UNet_pre_test=[
    "python", "src/inference.py",
    "--data_path", "dataset/oxford-iiit-pet",
    "--model","saved_models/UNetPre.pth",
    "--batch_size",'16',
    "--model_type", "UNet",
    '--data_preprocess','True',
    '--save_fig_filename', 'TestUNetPre.png'

]

ResNet34UNet_No_pre_test=[
    "python", "src/inference.py",
    "--data_path", "dataset/oxford-iiit-pet",
    "--model","saved_models/ResNet34UNetNoPre.pth",
    "--batch_size",'16',
    "--model_type", "ResNet34UNet",
    '--data_preprocess','False',
    '--save_fig_filename', 'TestResNet34UNetNoPre.png'

]


ResNet34UNet_pre_test=[
    "python", "src/inference.py",
    "--data_path", "dataset/oxford-iiit-pet",
    "--model","saved_models/ResNet34UNetPre.pth",
    "--batch_size",'16',
    "--model_type", "ResNet34UNet",
    '--data_preprocess','True',
    '--save_fig_filename', 'TestResNet34UNetPre.png'

]




subprocess.run(UNet_No_pre_cmd )
subprocess.run(UNet_pre_cmd)
subprocess.run(ResNet34UNet_No_pre_cmd )
subprocess.run(ResNet34UNet_pre_cmd )
subprocess.run(UNet_No_pre_test)
subprocess.run(UNet_pre_test)
subprocess.run(ResNet34UNet_No_pre_test)
subprocess.run(ResNet34UNet_pre_test)
