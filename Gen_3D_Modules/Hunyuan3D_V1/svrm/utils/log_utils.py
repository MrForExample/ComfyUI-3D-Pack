import cv2
import numpy as np

def txt_to_img(text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, img_width=1000, img_height=100, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    lines = text.split('\n')
    img_lines = []
    for line in lines:
        # 计算每行文本的尺寸
        line_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        line_width, line_height = line_size
        # 创建包含当前行的图像画布
        img_line = np.full((int(line_height*1.5) , img_width, 3), bg_color, dtype=np.uint8)
        text_x, text_y = 0, line_height
        cv2.putText(img_line, line, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        img_lines.append(img_line)


    # 垂直堆叠所有行图像
    img = np.vstack(img_lines)
    return img