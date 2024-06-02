from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from unet_scripts.model import UNET
from matplotlib import pyplot as plt
import cv2
import os
from PIL import Image
import numpy as np
from utils.watershed import get_instances
from skimage.transform import resize
from torchvision import transforms, models
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = int(1003 * 0.32)
IMAGE_WIDTH = int(1546 * 0.32)
# VAL_IMG_DIR = "/home/kashis/Desktop/Capstone/ridha_Unet/Dataset/val_images"
VAL_IMG_DIR = "/home/kashis/Desktop/Capstone/dataset/Pivot GIS Project/images_classification/tmp"
# VAL_IMG_DIR = "/home/kashis/Desktop/Capstone/dataset/may9_new_pivots/images"
unet_checkpoint_path = '/home/kashis/Desktop/Capstone/pipeline/pretrained_weights/my_check_may14.pth.tar'
classifier_checkpoint_path = '/home/kashis/Desktop/Capstone/CPIS_pipeline/cpis_angle_classifier/resnet101_custom_dataset_regression_retrained.pth'
OUT_DIR = "/home/kashis/Desktop/Capstone/pipeline/saved_images"

def plot_precision_recall_curve(y_true, y_pred_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_scores)
    average_precision = average_precision_score(y_true, y_pred_scores)

    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.show()
    plt.savefig('precision_recall_curve.png')
    
def get_bbox_from_seg_mask(seg_mask):
    seg_mask = np.array(seg_mask, dtype=int)
    a = np.where(seg_mask != 0)
    height = np.max(a[0]) - np.min(a[0])
    width = np.max(a[1]) - np.min(a[1])
    top_left = (np.min(a[1]), np.min(a[0]))
    return [top_left[0], top_left[1], width, height]

def main():

    transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    unet_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    unet_model.load_state_dict(torch.load(unet_checkpoint_path)["state_dict"])
    unet_model.eval()
    
    img_paths = os.listdir(VAL_IMG_DIR)
    for img_name in tqdm(img_paths):
       
        img = np.array(Image.open(os.path.join(VAL_IMG_DIR, img_name)).convert('RGB'))
        x = transforms(image=img) 
        x = x['image'].unsqueeze(0).to(DEVICE) 

        preds = torch.sigmoid(unet_model(x))
        preds = (preds > 0.5).float()

        pred = np.squeeze(preds, (0,1)).cpu()

        pred_np = pred.unsqueeze(-1).cpu().detach().numpy().astype(np.uint8)


        # INSTANCE SEGMENTATION
        rgb_np_pred = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2RGB)

        DIST_TRANSFORM_SCALE = 0.02
        MIN_SEG_AREA = 1200

        segments_in_img, seg_area = get_instances(rgb_np_pred, img_name, DIST_TRANSFORM_SCALE, MIN_SEG_AREA)
        combined_segments = np.sum(segments_in_img, axis=0)

        plt.figure(3)
        fig, ax = plt.subplots(1,3)
        fig.suptitle(img_name)
        ax[0].imshow(img)
        ax[1].imshow(rgb_np_pred[:, :, 0])
        ax[2].imshow(combined_segments)
        fig.show()
        # plt.close(fig) #689 698 697

        for seg_mask in segments_in_img:
            resized_seg_mask = resize(seg_mask, (1003, 1546), preserve_range=True)
            resized_seg_mask = resized_seg_mask.astype(seg_mask.dtype)
            [X, Y, H, W] = get_bbox_from_seg_mask(resized_seg_mask)
            seg_shape = resized_seg_mask.shape

            # Get a slightly bigger bbox
            clip_1 = max(int(Y-W/2), 0)
            clip_2 = min(int(Y+W+W/2), seg_shape[0])
            clip_3 = max(int(X-H/2), 0)
            clip_4 = min(int(X+H+H/2), seg_shape[1])

            cropped_image = img[clip_1:clip_2, clip_3:clip_4]

            if cropped_image.shape[0] == 0:
                continue

            # CLASSIFIER
            classifier_model = models.resnet101(pretrained=True)
            num_features = classifier_model.fc.in_features
            classes = [0, 180, 270, 360]
            num_classes = len(classes)
            classifier_model.fc = nn.Linear(num_features, 1)
            classifier_model.load_state_dict(torch.load(classifier_checkpoint_path))
            classifier_model.eval()
            classifier_model.to(DEVICE)
            x = transforms(image = cropped_image) 
            x = x['image'].unsqueeze(0).to(DEVICE)
            out = classifier_model(x)

            pred = out.detach().cpu().numpy()
            predicted = np.array(pred, dtype=int)

            print('Predicted angle: ', predicted)


            # _, predicted = torch.max(out.data, 1)
            # print('Angle classifier: ', img_name, classes[predicted.item()])

            # plt.figure(1)
            # plt.imshow(img)
            # plt.figure(2)
            # plt.imshow(cropped_image)
            # plt.show()

            # fig, ax = plt.subplots(1,4)
            # fig.suptitle(img_name)
            # ax[0].imshow(img)
            # ax[1].imshow(rgb_np_pred[:, :, 0])
            # ax[2].imshow(combined_segments)
            # ax[3].imshow(cropped_image[:, :, 0])
            # fig.show()
            # plt.close(fig)
            # print()

        # fig.savefig(os.path.join(OUT_DIR, img_name))



if __name__ == "__main__":
    main()
