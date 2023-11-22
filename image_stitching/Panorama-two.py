import os
import cv2
from Stitcher import Stitcher


if __name__ == '__main__':

    data_dir = 'data/Panorama-two/'

    for img in [['DSC02255.jpg', 'DSC02258.jpg'], ['IMG_0572.jpg', 'IMG_0573.jpg'], ['IMG_1325.jpg', 'IMG_1326.jpg']]:
        image_list =[]
        image_list.append(cv2.imread(os.path.join(data_dir, img[0])))
        image_list.append(cv2.imread(os.path.join(data_dir, img[1])))

        save_path = 'results/Panorama-two/'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        output = os.path.join(save_path, img[0])
        output_vis = os.path.join(save_path, img[0].split('.')[0] + '_vis.jpg')

        # 将图像拼接在一起以创建全景
        stitcher = Stitcher()
        result, vis = stitcher.stitch(image_list)

        # 显示图像
        cv2.imwrite(output, result)
        cv2.imwrite(output_vis, vis)
