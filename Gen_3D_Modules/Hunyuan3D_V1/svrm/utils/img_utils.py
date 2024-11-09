import os
import cv2
import numpy as np
from skimage.metrics import hausdorff_distance
from matplotlib import pyplot as plt


def get_input_imgs_path(input_data_dir):
    path = {}
    names = ['000', 'ori_000']
    for name in names:
        jpg_path = os.path.join(input_data_dir, f"{name}.jpg")
        png_path = os.path.join(input_data_dir, f"{name}.png")
        if os.path.exists(jpg_path):
            path[name] = jpg_path
        elif os.path.exists(png_path):
            path[name] = png_path
    return path


def rgba_to_rgb(image, bg_color=[255, 255, 255]):
    if image.shape[-1] == 3: return image
        
    rgba = image.astype(float)
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3] / 255.0

    bg = np.ones((image.shape[0], image.shape[1], 3), dtype=np.float32) 
    bg = bg * np.array(bg_color, dtype=np.float32)

    rgb = rgb * alpha[:, :, np.newaxis] + bg * (1 - alpha[:, :, np.newaxis])
    rgb = rgb.astype(np.uint8)
    
    return rgb


def resize_with_aspect_ratio(image1, image2, pad_value=[255, 255, 255]):
    aspect_ratio1 = float(image1.shape[1]) / float(image1.shape[0])
    aspect_ratio2 = float(image2.shape[1]) / float(image2.shape[0])

    top_pad, bottom_pad, left_pad, right_pad = 0, 0, 0, 0
    
    if aspect_ratio1 < aspect_ratio2:
        new_width = (aspect_ratio2 * image1.shape[0])
        right_pad = left_pad = int((new_width - image1.shape[1]) / 2)
    else:
        new_height = (image1.shape[1] / aspect_ratio2)
        bottom_pad = top_pad = int((new_height - image1.shape[0]) / 2)

    image1_padded = cv2.copyMakeBorder(
        image1, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=pad_value
    )
    return image1_padded


def estimate_img_mask(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用大津法进行阈值分割
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # mask_otsu = thresh.astype(bool)
    # thresh_gray = 240

    # 使用 Canny 边缘检测算法找到边缘
    edges = cv2.Canny(gray, 20, 50)

    # 使用形态学操作扩展边缘
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空的 mask
    mask = np.zeros_like(gray, dtype=np.uint8)

    # 根据轮廓信息填充 mask（使用 thickness=cv2.FILLED 参数）
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    mask = mask.astype(bool)

    return mask


def compute_img_diff(img1, img2, matches1, matches1_from_2, vis=False):
    scale = 0.125
    gray_trunc_thres = 25 / 255.0

    # Match
    if matches1.shape[0] > 0:
        match_scale = np.max(np.ptp(matches1, axis=-1))
        match_dists = np.sqrt(np.sum((matches1 - matches1_from_2) ** 2, axis=-1))
        dist_threshold = match_scale * 0.01
        match_num = np.sum(match_dists <= dist_threshold)
        match_rate = np.mean(match_dists <= dist_threshold)
    else:
        match_num = 0
        match_rate = 0

    # IOU
    img1_mask = estimate_img_mask(img1)
    img2_mask = estimate_img_mask(img2)
    img_intersection = (img1_mask == 1) & (img2_mask == 1)
    img_union = (img1_mask == 1) | (img2_mask == 1)
    intersection = np.sum(img_intersection == 1)
    union = np.sum(img_union == 1)
    mask_iou = intersection / union if union != 0 else 0

    # Gray
    height, width = img1.shape[:2]
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.GaussianBlur(img1_gray, (7, 7), 0)
    img2_gray = cv2.GaussianBlur(img2_gray, (7, 7), 0)

    # Gray Diff
    img1_gray_small = cv2.resize(img1_gray, (int(width * scale), int(height * scale)),
                                 interpolation=cv2.INTER_LINEAR) / 255.0
    img2_gray_small = cv2.resize(img2_gray, (int(width * scale), int(height * scale)),
                                 interpolation=cv2.INTER_LINEAR) / 255.0
    img_gray_small_diff = np.abs(img1_gray_small - img2_gray_small)
    gray_diff = img_gray_small_diff.sum() / (union * scale) if union != 0 else 1

    img_gray_small_diff_trunc = img_gray_small_diff.copy()
    img_gray_small_diff_trunc[img_gray_small_diff < gray_trunc_thres] = 0
    gray_diff_trunc = img_gray_small_diff_trunc.sum() / (union * scale) if union != 0 else 1

    # Edge
    img1_edge = cv2.Canny(img1_gray, 100, 200)
    img2_edge = cv2.Canny(img2_gray, 100, 200)
    bw_edges1 = (img1_edge > 0).astype(bool)
    bw_edges2 = (img2_edge > 0).astype(bool)
    hausdorff_dist = hausdorff_distance(bw_edges1, bw_edges2)
    if vis == True:
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(img1_gray, cmap='gray')
        axs[0].set_title('Img1')
        axs[1].imshow(img2_gray, cmap='gray')
        axs[1].set_title('Img2')
        axs[2].imshow(img1_mask)
        axs[2].set_title('Mask1')
        axs[3].imshow(img2_mask)
        axs[3].set_title('Mask2')
        plt.show()
        plt.figure()
        mask_cmp = np.zeros((height, width, 3))
        mask_cmp[img_intersection, 1] = 1
        mask_cmp[img_union, 0] = 1
        plt.imshow(mask_cmp)
        plt.show()
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(img1_gray_small, cmap='gray')
        axs[0].set_title('Img1 Gray')
        axs[1].imshow(img2_gray_small, cmap='gray')
        axs[1].set_title('Img2 Gary')
        axs[2].imshow(img_gray_small_diff, cmap='gray')
        axs[2].set_title('diff')
        axs[3].imshow(img_gray_small_diff_trunc, cmap='gray')
        axs[3].set_title('diff_trunct')
        plt.show()
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(img1_edge, cmap='gray')
        axs[0].set_title('img1_edge')
        axs[1].imshow(img2_edge, cmap='gray')
        axs[1].set_title('img2_edge')
        plt.show()

    info = {}
    info['match_num'] = match_num
    info['match_rate'] = match_rate
    info['mask_iou'] = mask_iou
    info['gray_diff'] = gray_diff
    info['gray_diff_trunc'] = gray_diff_trunc
    info['hausdorff_dist'] = hausdorff_dist
    return info


def predict_match_success_human(info):
    match_num = info['match_num']
    match_rate = info['match_rate']
    mask_iou = info['mask_iou']
    gray_diff = info['gray_diff']
    gray_diff_trunc = info['gray_diff_trunc']
    hausdorff_dist = info['hausdorff_dist']

    if mask_iou > 0.95:
        return True

    if match_num < 20 or match_rate < 0.7:
        return False

    if mask_iou > 0.80 and gray_diff < 0.040 and gray_diff_trunc < 0.010:
        return True

    if mask_iou > 0.70 and gray_diff < 0.050 and gray_diff_trunc < 0.008:
        return True

    '''
    if match_rate<0.70 or match_num<3000:
        return False

    if (mask_iou>0.85 and hausdorff_dist<20)or (gray_diff<0.015 and gray_diff_trunc<0.01) or match_rate>=0.90:
        return True
    '''

    return False


def predict_match_success(info, model=None):
    if model == None:
        return predict_match_success_human(info)
    else:
        feat_name = ['match_num', 'match_rate', 'mask_iou', 'gray_diff', 'gray_diff_trunc', 'hausdorff_dist']
        # 提取特征
        features = [info[f] for f in feat_name]
        # 预测
        pred = model.predict([features])[0]
        return pred >= 0.5