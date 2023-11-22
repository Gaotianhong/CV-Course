import cv2
import numpy as np

from copy import deepcopy


class Stitcher:

    def get_homography(self, imageA, imageB, ratio=0.75, reprojThresh=4.0):
        # 解压缩图像，然后从它们中检测关键点以及提取局部不变描述符（SIFT）

        (kpsA, featuresA) = self.detectAndDescribe(imageA)

        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # 匹配两幅图像之间的特征
        M = self.matchKeypoints(kpsA, kpsB, featuresA,
                                featuresB, ratio, reprojThresh)

        # 如果匹配结果M返回空，表示没有足够多的关键点匹配信息去创建一副全景图
        if M is None:
            return None

        # 若M不为None，则使用透视变换来拼接图像
        (matches, H, status) = M
        return H, kpsA, kpsB, matches, status

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=True):

        (imageB, imageA) = images

        H, kpsA, kpsB, matches, status = self.get_homography(
            imageA, imageB, ratio, reprojThresh)

        result = np.ones(
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0], 3))

        result = cv2.warpPerspective(
            imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]), borderValue=(255, 255, 255))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检查是否应该可视化关键点匹配
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                                   status)

            # 返回拼接图像的元组和可视化
            return result, vis

        # 返回拼接图像
        return result

    def detectAndDescribe(self, image):
        # 从图像中检测并提取特征
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将关键点从KeyPoint对象转换为NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        keypoint_image = deepcopy(image)
        for point in kps:
            point = np.array(point, dtype=int)
            cv2.circle(keypoint_image, point, 2, (0, 0, 255), 3)

        cv2.imwrite('result/SIFT Key Points.jpg', keypoint_image)

        # 返回关键点和特征的元组
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # 计算原始匹配项并初始化实际匹配项列表
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # 循环原始匹配
        for m in rawMatches:
            # 确保距离在一定的比例内(即 Lowe's ratio)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 计算单应性至少需要4个匹配项
        if len(matches) > 4:
            # 构造两组点
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算两组点之间的单应性
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # 返回匹配以及单应矩阵和每个匹配点的状态
            return (matches, H, status)

        # 否则，将无法计算单应性
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.ones((max(hA, hB), wA + wB, 3), dtype="uint8")*255
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
