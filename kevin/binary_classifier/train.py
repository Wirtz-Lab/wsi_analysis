#%%
#import necessary modules:
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from natsort import natsorted
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
import timm
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import random
import time
import gc
from sklearn.model_selection import StratifiedKFold
import copy
from collections import defaultdict
from sklearn.metrics import f1_score
#%%
#define dataset and dataloaders
train_df_src = r'\\fatherserverdw\Kevin\unstained_blank_classifier\train_df.xlsx'
train_df = pd.read_excel(train_df_src) # 1= white , 0=nonwhite, unbalanced, 79271 0's and 195376 1's. Need stratifiedgroupKfold for CV.
train_df = train_df.drop(columns="Unnamed: 0")
train_df
#%% md
### First find mean and std of dataset for image normalization:
### code for finding dataset std and mean from: https://kozodoi.me/blog/20210308/compute-image-stats
#%%
find_mean_std_dataset = False # already found it

if find_mean_std_dataset:
    class Unstain2StainData(Dataset):
        def __init__(self,df,transform=None):
            self.df = df
            self.directory = df["imagepath"].tolist()
            self.transform = transform

        def __len__(self):
            return int(len(self.directory)/3)

        def __getitem__(self,idx):
            path = self.directory[idx]
            image = cv2.imread(path, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                image = self.transform(image = image)['image']
            return image

    device      = torch.device('cpu')
    num_workers = 0
    image_size  = 384
    batch_size  = 4

    augmentations = A.Compose([A.Resize(height= image_size ,width = image_size ),
                                       A.Normalize(mean=(0,0,0), std=(1,1,1)),
                                       ToTensorV2()])

    unstain2stain_dataset = Unstain2StainData(df = train_df, transform = augmentations)# data loader
    image_loader = DataLoader(unstain2stain_dataset,
                              batch_size  = batch_size,
                              shuffle     = False,
                              num_workers = num_workers,
                              pin_memory  = True)

    images = next(iter(image_loader))
    print("Images have a tensor size of {}.".
          format(images.size()))
    # compute mean/std for 1/3 of the images for time's sake:
    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader,colour='red'):
        psum    += inputs.sum(axis = [0, 2, 3]) # sum over axis 1
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3]) # sum over axis 1

    # pixel count
    count = len(train_df) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))

    #mean=[0.2966, 0.3003, 0.3049], std=[0.4215, 0.4267, 0.4332]
#%% md
### We can now use the above calculated mean and std value for our augmentations for the dataset. Also, use StratifiedKfold to perform 5-fold CV with each fold containing the equal percentages of the blank and non-blank images:

#%%
# add stratifiedkfold to df:
new_df_train = train_df.copy(deep=True)
strat_kfold = StratifiedKFold(shuffle = True, random_state = 42) #use default n_split = 5, random_state for reproducibility

#split on white and non-white and add a new column fold to it:
for each_fold, (idx1,idx2) in enumerate (strat_kfold.split(X = new_df_train, y = new_df_train['label'])):
    new_df_train.loc[idx2,'fold'] = int(each_fold) #create new fold column with the fold number (up to 5)

new_df_train["fold"] = new_df_train["fold"].apply(lambda x: int(x)) # somehow doesn't turn to int, so change to int, fold from 0~4
#%%
new_df_train
#%%
#check if stratification worked by grouping:
grouped = new_df_train.groupby(['fold','label']) # look how it's splitted
print(grouped.fold.count())

ratio_list = []
for k in range(5):
    ratio = grouped.fold.count()[k][0]/grouped.fold.count()[k][1]
    ratio_list.append(ratio)
print("the ratios of the folds are: {}".format(ratio_list)) #ratios to check stratification
#%% md
### As we can see above, stratification was successful. Now define transforms:
#%%
#define transforms/image augmentation for the dataset

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(384), # efficientnetv2_s 384 x 384
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2966, 0.3003, 0.3049], std=[0.4215, 0.4267, 0.4332]) #calculated above mean & std
])

val_transform = transforms.Compose([
 # validate at 1024 x 1024, you want to use val dataset to real world application, but maybe resize to 384 if performance is bad.
    #transforms.Resize(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2966, 0.3003, 0.3049], std=[0.4215, 0.4267, 0.4332]) #calculated above mean & std
])
#%%
# build train dataset
class TrainDataSet(Dataset):
    # initialize df, label, imagepath and transforms
    def __init__(self, df, label=True, transforms = None):
        self.df = df
        self.label = df["label"].tolist()
        self.imagepaths = df["imagepath"].tolist()
        self.transforms = transforms
    # define length, which is simply length of all imagepaths
    def __len__(self):
        return len(self.df)
    # define main function to read image and label, apply transform function and return the transformed images.
    def __getitem__(self,idx):
        image_path = self.imagepaths[idx]
        img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.label:
            label = self.label[idx]
        if self.transforms is not None:
            image = self.transforms(img)

        return image, label
#%%
# all model configs go here so that they can be changed when we want to:
class model_config:
    seed = 42
    model_name = "efficientnetv2_l"
    train_batch_size = 16
    valid_batch_size = 32
    epochs = 5
    learning_rate = 0.001
    scheduler = "CosineAnnealingLR"
    T_max = int(30000/train_batch_size*epochs) # for cosineannealingLR, explore different values
    weight_decay = 1e-6 # explore different weight decay (Adam optimizer)
    n_accumulate = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iters_to_accumulate = max(1,32//train_batch_size) # for scaling accumulated gradients
    eta_min = 1e-5
    model_save_directory = os.path.join(os.getcwd(),"kevin","binary_classifier","outputs","model") #assuming os.getcwd is the wsi_analysis directory
#%%
# sets the seed of the entire notebook so results are the same every time we run for reproducibility. no randomness, everything is controlled.
def set_seed(seed = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(model_config.seed)
#%%
# define dataloading function:
def load_dataset(fold):
    model_df_train = new_df_train.query("fold!=@fold").reset_index(drop=True)
    model_df_val = new_df_train.query("fold==@fold").reset_index(drop=True)
    train_dataset = TrainDataSet(df = model_df_train, transforms = train_transform)
    val_dataset = TrainDataSet(df = model_df_val, transforms = val_transform)
    train_dataloader = DataLoader(dataset = train_dataset,
        batch_size = model_config.train_batch_size, # pin_memory= true allows faster data transport from cpu to gpu
        num_workers = 0, pin_memory = True, shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset,
        batch_size = model_config.valid_batch_size,
        num_workers = 0, pin_memory = True, shuffle = True)
    return train_dataloader, val_dataloader
#%%
train_dataloader, val_dataloader = load_dataset(fold = 0)
image, labels = next(iter(train_dataloader))
print("Images have a tensor size of {}, and Labels have a tensor size of {}".
      format(image.size(),labels.size()))
#%%
images, labels = next(iter(val_dataloader))
print("Images have a tensor size of {}, and Labels have a tensor size of {}".
      format(images.size(),labels.size()))
#%% md
### Now move on to the model and loss function:
#%%
def build_model():
    model = timm.create_model(model_config.model_name,pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features,1) #in_features = 1280, out_features = 1, so that 0 or 1 binary classification
    # model.add_module('sigmoid', nn.Sigmoid()) # obtain probability b/w 0 and 1
    model.to(model_config.device) # model to gpu
    return model
#%%
def loss_func(y_pred,y_true):
    loss = nn.BCEWithLogitsLoss()
    return loss(y_pred,y_true)
#%%
def return_f1_score(y_true,y_pred):
    # y_true = y_true.to(torch.int)
    # y_pred = y_pred.to(torch.int)
    f_one_score = f1_score(y_true,y_pred)
    return f_one_score
#%% md
### Training Loop:
#%%
#from: https://pytorch.org/docs/stable/notes/amp_examples.html in "working with scaled gradients"
def epoch_train(model, optimizer, scheduler, dataloader, device, epoch):
    model.train() # set mode to train
    dataset_size = 0 #initialize
    running_loss = 0.0 #initialize
    scaler = GradScaler() # enable GradScaler
    pbar = tqdm(enumerate(dataloader), total = len(dataloader), desc='Train')
    for idx, (images, labels) in pbar:
        images = images.to(device, dtype=torch.float) # move tensor to gpu
        labels  = labels.to(device, dtype=torch.float) # move tensor to gpu
        batch_size = images.size(0) # return batch size (N*C*H*W), or N (64)

        with autocast(enabled=True,dtype=torch.float16): # enable autocast for forward pass
            y_pred = model(images) # forward pass, get y_pred from input
            labels = labels.unsqueeze(1) #make tensor size from [16] to [16,1] for label, same as images
            loss   = loss_func(y_pred, labels) # compute losses from y_pred
            loss   = loss / model_config.iters_to_accumulate # need to normalize since accumulating gradients
        scaler.scale(loss).backward() # accumulates the scaled gradients

        if (idx + 1) % model_config.iters_to_accumulate == 0: # scale updates should only happen at batch granularity
            scaler.step(optimizer)
            scaler.update() # update scale for next iteration
            optimizer.zero_grad() # zero the accumulated scaled gradients
            scheduler.step() # change lr,make sure to call this after scaler.step

        running_loss += (loss.item() * batch_size) # update current running loss for all images in batch
        dataset_size += batch_size # update current datasize

        epoch_loss = running_loss / dataset_size # get current epoch average loss
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}')

    torch.cuda.empty_cache() #clear gpu memory after every epoch
    gc.collect()

    return epoch_loss #return loss for this epoch
#%% md
### Validation Loop:
#%%
@torch.no_grad() # disable gradient calc for validation
def epoch_valid(model, dataloader, device, epoch):
    model.eval() # set mode to eval
    dataset_size = 0 #initialize
    running_loss = 0.0 #initialize
    valid_score_history = [] #keep validation score
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation')
    for idx, (images, labels) in pbar:
        images  = images.to(device, dtype=torch.float)
        labels   = labels.to(device, dtype=torch.float)
        batch_size = images.size(0)
        y_pred  = model(images)
        labels = labels.unsqueeze(1)
        loss    = loss_func(y_pred, labels)

        running_loss += (loss.item() * batch_size) #update current running loss
        dataset_size += batch_size #update current datasize
        epoch_loss = running_loss / dataset_size #divide epoch loss by current datasize

        y_pred = nn.Sigmoid()(y_pred) #sigmoid for binary classification
        labels = labels.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.round(y_pred)
        f_one_score = return_f1_score(labels,y_pred) #fetch f1 score
        valid_score_history.append(f_one_score)

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.3f}',
                        lr=f'{current_lr:0.4f}')
    valid_score_history = np.mean(valid_score_history, axis=0)
    torch.cuda.empty_cache() #clear gpu memory after every epoch
    gc.collect()

    return epoch_loss, valid_score_history #return loss and valid_score_history for this epoch

#%% md
### Run Training:
#%%
def run_training(model, optimizer, scheduler, device, num_epochs):

    start = time.time() # measure time
    best_model_wts = copy.deepcopy(model.state_dict()) #deepcopy
    best_f1      = 0 # initial best score
    best_epoch     = -1 # initial best epoch
    history = defaultdict(list) # history defaultdict to store relevant variables

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')
        train_loss = epoch_train(model, optimizer, scheduler,
                                           dataloader=train_dataloader,
                                           device=model_config.device, epoch=epoch)
        valid_loss, valid_score_history = epoch_valid(model, val_dataloader,
                                                 device=model_config.device,
                                                 epoch=epoch)
        valid_f1 = valid_score_history
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(valid_loss)
        history['Valid Dice'].append(valid_f1)

        print(f'Valid Dice: {valid_f1:0.4f}')

        # if dice score improves, save the best model
        if valid_f1 >= best_f1:
            print(f"Validation F1-Score Improved ({best_f1:0.4f} ---> {valid_f1:0.4f})")
            best_f1    = valid_f1
            best_epoch   = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = os.path.join(model_config.model_save_directory,f"best_epoch-{fold:02d}.pt")
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            torch.save(model.state_dict(), PATH)
            print("Model Saved!")

        # save the most recent model
        latest_model_wts = copy.deepcopy(model.state_dict())
        PATH = os.path.join(model_config.model_save_directory,f"latest_epoch-{fold:02d}.pt")
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        torch.save(model.state_dict(), PATH)

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60))
    print("Best F1-Score: {:.4f}".format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
#%% md
### Now build model using functions and train:
#%%
model = build_model()
optimizer = optim.Adam(model.parameters(),
                       lr=model_config.learning_rate,
                       weight_decay = model_config.weight_decay ) # default learning rate
if model_config == "CosineAnnealingLR": # change to CosineAnnealingLR
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = model_config.T_max,
                                               eta_min =  model_config.eta_min)
#%%
# Run Training!
for fold in range(4):
    print(f'Fold: {fold}')
    train_dataloader, valid_dataloader = load_dataset(fold = fold)
    model     = build_model()
    optimizer = optim.Adam(model.parameters(),
                           lr=model_config.learning_rate,
                           weight_decay=model_config.weight_decay) # default learning rate

    if model_config.scheduler == "CosineAnnealingLR": # change to CosineAnnealingLR
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max = model_config.T_max,
                                                   eta_min =  model_config.eta_min)

    model, history = run_training(model, optimizer, scheduler,
                                  device=model_config.device,
                                  num_epochs=model_config.epochs)
#%%
