
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
import numpy as np
import skimage.draw
from tqdm import tqdm


DATA_DIR = '/home/kashis/Desktop/Capstone/dataset/may9_new_pivots'
MASK_OUT_DIR = '/home/kashis/Desktop/Capstone/dataset/may9_new_pivots/masks'



for file_name in tqdm(sorted(os.listdir(DATA_DIR))):
    cpis_name = file_name.split('.')[0]
    if file_name.endswith('.png'):

        img = cv2.imread(os.path.join(DATA_DIR, file_name))
        
        pkl_fil_path = cpis_name + '.pkl'

        print(file_name)
        pkl_file = open(os.path.join(DATA_DIR, pkl_fil_path), 'rb')
        pivot_infos = pickle.load(pkl_file)

        ht, wt, _ = img.shape
        seg_mask = np.zeros([ht, wt], dtype=np.uint8)
        fig, ax = plt.subplots(1)

        for pivot in pivot_infos:
            x, y, r = pivot['pixel_x'], pivot['pixel_y'], pivot['pixel_rad']

            circ = Circle((x,y), r)
            ax.add_patch(circ)

            rr, cc = skimage.draw.disk((int(y), int(x)), r)
            
            rr_temp = []
            cc_temp = []
            for r,c in zip(rr, cc):
                if r>=0 and r < ht and c>=0 and c < wt:
                    rr_temp.append(r)
                    cc_temp.append(c)


            
            seg_mask[rr_temp, cc_temp] = 1

        plt.imshow(img, cmap='gray')
        plt.imshow(seg_mask, cmap='jet', alpha=0.5) 
        plt.show()
        plt.close()
        
        #plt.imsave(os.path.join(MASK_OUT_DIR, file_name), seg_mask, cmap='gray')
