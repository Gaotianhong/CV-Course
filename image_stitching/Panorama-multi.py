import os
import cv2
import imutils
import numpy as np
from imutils import paths


if __name__ == '__main__':

    data_name = 'Xue-Mountain-Enterance'

    # 获取图像列表
    image_folder = 'data/Panorama-multi/{}'.format(data_name)
    crop = 0

    save_path = 'results/Panorama-multi/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    output = 'results/Panorama-multi/{}.png'.format(data_name)

    imagePaths = sorted(list(paths.list_images(image_folder)))
    images = []

    # 加载图像
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)

    # 拼接
    print("[INFO] stitching images...")
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)

    if status == 0:
        # status=0时，表示拼接成功
        if crop > 0:
            # 边界填充
            stitched = cv2.copyMakeBorder(
                stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

            # 转为灰度图进行并二值化
            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

            # 获取轮廓
            cnts = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            mask = np.zeros(thresh.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            minRect = mask.copy()
            sub = mask.copy()

            while cv2.countNonZero(sub) > 0:
                minRect = cv2.erode(minRect, None)
                sub = cv2.subtract(minRect, thresh)

            cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)

            # 取出图像区域
            stitched = stitched[y:y + h, x:x + w]

        cv2.imwrite(output, stitched)
    else:
        print("[INFO] image stitching failed ({})".format(status))
