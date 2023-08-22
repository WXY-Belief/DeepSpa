import numpy as np
import cv2
import os
from tqdm import tqdm

if __name__ == '__main__':
    input_img = cv2.imread("DAPI.tif", cv2.IMREAD_GRAYSCALE)
    h, w = input_img.shape[:2]
    new_h = int(np.ceil(h / 512) * 512)
    new_w = int(np.ceil(w / 512) * 512)
    grid_h = int(new_h / 512)
    grid_w = int(new_w / 512)
    img_padded = np.zeros((new_h, new_w), dtype=np.uint8)
    img_padded[:h, :w] += input_img
    os.makedirs("split_20", exist_ok=True)
    for i in tqdm(range(grid_h)):
        for j in range(grid_w):
            crop = img_padded[i * 512:i * 512 + 512, j * 512: j * 512 + 512]
            crop[crop < 20] = 0
            crop[crop >= 20] -= 20
            cv2.imwrite(
                "split_20/DAPI_%03d_%03d.png" % (i, j),
                crop
            )
