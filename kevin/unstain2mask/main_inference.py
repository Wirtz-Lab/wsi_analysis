#%%
# first import all of the packages required in this entire project:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from glob import glob
import copy
import joblib
from tqdm import tqdm

tqdm.pandas()
import gc
from collections import defaultdict
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
import cv2
import matplotlib

matplotlib.style.use('ggplot')
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

Image.MAX_IMAGE_PIXELS = None
pd.set_option('display.float_format', '{:.2f}'.format)
import segmentation_models_pytorch as smp
from sklearn.preprocessing import KBinsDiscretizer
from torchvision.transforms import functional as F
#%%

# all model configs go here so that they can be changed when we want to:
class model_config:
    seed = 42
    encoder_name = "timm-resnest200e"  #"efficientnet-b4" # "tu-efficientnetv2_m" # from https://smp.readthedocs.io/en/latest/encoders_timm.html
    train_batch_size = 4
    valid_batch_size = 4
    infer_batch_size = 2
    epochs = 5
    learning_rate = 0.001
    scheduler = "CosineAnnealingLR"
    T_max = int(30000 / train_batch_size * epochs)  # for cosineannealingLR, explore different values
    weight_decay = 1e-6  # explore different weight decay (Adam optimizer)
    n_accumulate = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iters_to_accumulate = max(1, 32 // train_batch_size)  # for scaling accumulated gradients
    eta_min = 1e-5
    model_save_directory = os.path.join(os.getcwd(), "model",
                                        str(encoder_name))  #assuming os.getcwd is the current training script directory


# sets the seed of the entire notebook so results are the same every time we run for reproducibility. no randomness, everything is controlled.
def set_seed(seed=42):
    np.random.seed(seed)  #numpy specific random
    random.seed(seed)  # python specific random (also for albumentation augmentations)
    torch.manual_seed(seed)  # torch specific random
    torch.cuda.manual_seed(seed)  # cuda specific random
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # when deterministic = true, benchmark = False, otherwise might not be deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)  # set a fixed value for the hash seed, for hases like dictionary


set_seed(model_config.seed)
val_transform = A.Compose([
    # validate at 1024 x 1024, you want to use val dataset to real world application, but maybe resize to 384 if performance is bad.
    #A.Resize(384),
    ToTensorV2(),
    #A.Normalize(mean=(0.8989, 0.9101, 0.9236), std=(0.0377, 0.0389, 0.0389)) #calculated above mean & std
])
mean = torch.tensor([0.8989, 0.9106, 0.9245])
std = torch.tensor([0.0393, 0.0389, 0.0409])
#%%
# define the normalization function
def normalize(image):
    image = image.float() / 255.0  # convert image to float and scale to [0, 1]
    image = (image - mean[:, None, None]) / std[:, None,
                                            None]  # normalize each channel, where mean std is torch.Size[3,1,1] and image size is [3,512,512]
    return image

# build test dataset
class TestDataSet(Dataset):
    # initialize df, label, imagepath and transforms
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.imagepaths = df["image_path"].tolist()
        self.maskpaths = df["mask_path"].tolist()
        self.transforms = transforms

    # define length, which is simply length of all imagepaths
    def __len__(self):
        return len(self.df)

    # define main function to read image and label, apply transform function and return the transformed images.
    def __getitem__(self, idx):
        image_path = self.imagepaths[idx]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        # image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.label:
            mask_path = self.maskpaths[idx]
            # mask = Image.open(mask_path).convert("L")
            mask = cv2.imread(mask_path, 0)
            mask = np.array(mask)
            mask_12ch = np.zeros((1024, 1024, 12), dtype=np.float32)
            for class_idx in range(1, 13):
                class_pixels = (np.array(mask) == class_idx)
                mask_12ch[:, :, class_idx - 1] = class_pixels.astype(np.float32)
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask_12ch)
            image = transformed['image']
            image = normalize(image)
            mask = transformed['mask']
            mask = torch.permute(mask, (2, 0, 1))  # 512 x 512 x 12 -> 12 x 512 x 512 so it becomes N x C x H x W
        image = image.float()
        image = F.normalize(image, mean=(0.8989, 0.9101, 0.9236), std=(0.0377, 0.0389, 0.0389))
        return image, mask, mask_path  # return tensors of image arrays, image should be 1024 x 1024 x 3, mask 1024 x 1024, image_path just a list of image paths
#%%
# load test_df in from directory (includes images to run inference and ground truth mask path):
test_df_src = r"\\shelter\Kyu\unstain2mask\main\test_df_short.xlsx"
test_df = pd.read_excel(test_df_src)
test_df
#%%
# define dataloading function:
test_dataset = TestDataSet(df=test_df, transforms=val_transform)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=model_config.infer_batch_size,
                             num_workers=0, pin_memory=True, shuffle=True)

image, masks, mask_path = next(iter(test_dataloader))
print("Images have a tensor size of {}, Masks have a tensor size of {}, and Mask Paths have a length of {}".
      format(image.size(), masks.size(), len(mask_path)))
#%%
def build_model():
    model = smp.UnetPlusPlus(encoder_name=model_config.encoder_name, encoder_weights="imagenet", activation=None,
                             in_channels=3, classes=12, decoder_use_batchnorm=True)
    model.to(model_config.device)  # model to gpu
    return model


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    """
    Function to calculate dice coef, both y_true and y_pred needs to be of tensor size N x 12 x 1024 x 1024 (batch size x num_classes x image height x image width).
    """
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)  # binary tensor
    intersection = (y_true * y_pred).sum(dim=dim)  # calculate overlapping pixels b/w pred and true for height and width
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)  # denominator, A+B along height and width
    dice = ((2 * intersection + epsilon) / (den + epsilon)).mean(
        dim=(1, 0))  # avg over batch & channel to return scalar
    return dice


### Run inference, predicted mask tensor should be saved as the original mask array (take argmax along axis = 0). Dice score should be calculated b/w  return list of arrays size of batch size.
@torch.no_grad()
def infer(model_paths, test_loader, thr=0.5):
    model = build_model()  #initialize the model outside the loop
    pred_labels = []
    mask_names = []
    pred_labels_raw = []  # 12-channel one, returned so that dice score can be calculated after inference
    # dice_scores = []
    for idx, (image, masks, mask_path) in enumerate(tqdm(test_loader, total=len(test_loader), desc='Inference')):
        y_pred_list = []
        image = image.to(model_config.device, dtype=torch.float)
        masks = masks.to(model_config.device, dtype=torch.float)
        mask_names.append(mask_path)
        # y_pred = torch.zeros((model_config.infer_batch_size,12,1024,1024), device = model_config.device)# empty N x 12 x 1024 x 1024 tensor reinitialized for every batch
        # for path in model_paths:  # load five best models from each fold
        model.load_state_dict(torch.load(model_paths))
        model.eval()  # change model to eval stage
        output = model(
            image)  # output of size N x 12 x 1024 x 1024, each channel in dim=1 with 12 classes has logits of label for that specific pixel
        output = nn.Sigmoid()(
            output)  #make the output to 0~1 probabilities, model last layer doesn't have sigmoid in it, so this becomes probabilities
        y_pred_list.append(output)  #ensemble, append the y_preds as a list
        y_pred = torch.stack(y_pred_list, dim=0)  # stack the list along a fifth dimension
        y_pred = torch.mean(y_pred,
                            dim=0)  #ensemble, average the probabilities along the 5-th dimension to make it 4d again
        # calculate dice score before making 12-channel to 1-channel
        # dice_score = dice_coef(masks,y_pred) # dice score should be scalar, size BS x 1
        pred_labels_raw.append(y_pred)  # save the 12-channel pred labels
        y_pred = torch.argmax(y_pred,
                              dim=1)  # argmax along channel axis, returns the highest probable channel for that pixel
        y_pred = y_pred
        pred_labels.append(y_pred)  # list of arrays
        # dice_scores.append(dice_score) # list of dice scores
    return pred_labels_raw, pred_labels, mask_names  #,dice_scores
#%%
# Evaluate dice score save the array of arrays as an image at an directory :
import glob
# saved_model_path = model_config.model_save_directory
model_paths = r"C:\Users\labadmin\PycharmProjects\wsi_analysis\kevin\unstain2mask\model\timm-resnest200e\best_epoch-00.pt" #glob(f'{saved_model_path}/best_epoch*.pt') # try one model, keep getting cuda out of memory
print("Model list are {}".format(model_paths))
pred_labels_raw, pred_labels, mask_names = infer(model_paths, test_dataloader)
#%%
def convert_mask_to_binary_channel(mask_path):
    mask = cv2.imread(mask_path, 0)
    mask = np.array(mask)
    mask_12ch = np.zeros((1024, 1024, 12), dtype=np.float32)
    for class_idx in range(1, 13):
        class_pixels = (np.array(mask) == class_idx)
        mask_12ch[:, :, class_idx - 1] = class_pixels.astype(np.float32)
    mask_12ch = np.transpose(mask_12ch, (2, 0, 1))
    mask_12ch = np.expand_dims(mask_12ch, axis=0)
    return mask_12ch  # needs to be 1 x 12 x 1024 x 1024 (1 x num_classes x H x W)

def save_inference_results(pred_labels_raw, pred_labels, mask_names, save_dst_pth):
    results_df = pd.DataFrame(columns=["dice_score", "inference_mask_save_path", "true_mask_path"])
    pred_labels_raw_numpy = [x.cpu().detach().numpy() for x in pred_labels_raw]
    pred_labels_numpy = [x.cpu().detach().numpy() for x in pred_labels]  # list of pred_labels
    mask_names_numpy = [x for x in mask_names]
    dice_scores = []
    true_mask_names = []
    inference_save_paths = []
    for num_batch_idx in range(int(test_df.shape[
                                       0] / model_config.infer_batch_size)):  # length of test_df must be divisible by infer_batch_size
        batches_pred_labels_raw = pred_labels_raw_numpy[num_batch_idx]
        batches_pred_labels = pred_labels_numpy[num_batch_idx]
        batches_mask_names = mask_names_numpy[num_batch_idx]
        for batch_idx in range(model_config.infer_batch_size):
            each_pred_labels_raw = batches_pred_labels_raw[batch_idx]  # 12 x 1024 x 1024
            each_pred_labels_raw = np.expand_dims(each_pred_labels_raw, axis=0)  # 1 x 12 x 1024 x 1024 for dice score
            each_pred_label = batches_pred_labels[batch_idx]
            each_pred_label = each_pred_label.astype('uint8')
            each_mask_name = batches_mask_names[batch_idx]
            true_mask_names.append(each_mask_name)
            each_mask_ext = os.path.basename(each_mask_name)  #get only name to save!
            y_true = convert_mask_to_binary_channel(each_mask_name)
            y_true = torch.from_numpy(y_true)
            each_pred_labels_raw = torch.from_numpy(each_pred_labels_raw)
            dice_score = dice_coef(y_true, each_pred_labels_raw)
            dice_score = dice_score.cpu().detach().numpy()
            dice_scores.append(dice_score)
            img_save_pth = os.path.join(save_dst_pth, each_mask_ext)
            inference_save_paths.append(img_save_pth)
            Image.fromarray(each_pred_label).save(img_save_pth)
    results_df["dice_score"] = dice_scores
    results_df["inference_mask_save_path"] = inference_save_paths
    results_df["true_mask_path"] = true_mask_names
    return results_df
#%%
results_df = save_inference_results(pred_labels_raw, pred_labels, mask_names,
                                    r"\\shelter\Kyu\unstain2mask\main\inference")
results_df
#%%
save_src = r"\\shelter\Kyu\unstain2mask\main\inference\results_df.xlsx"
results_df.to_excel(save_src)