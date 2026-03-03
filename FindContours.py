# 获取固体内轮廓
import skimage.measure as measure
import skimage.morphology as mophology
import cv2
import numpy as np
import math

def draw_dashed_circle(img, center, radius, color, thickness=1, dash_length=10, gap_length=5):
    """
    绘制虚线圆
    
    参数:
        img: 图像数组
        center: 圆心坐标 (x, y)
        radius: 半径
        color: 颜色 (B, G, R)
        thickness: 线宽
        dash_length: 每段实线的长度（像素）
        gap_length: 每段间隔的长度（像素）
    """
    x, y = center
    # 计算圆的周长
    circumference = 2 * math.pi * radius
    # 计算需要多少段（实线+间隔）
    segment_length = dash_length + gap_length
    num_segments = int(circumference / segment_length)
    
    # 如果圆太小，直接画实线圆
    if num_segments < 8:
        cv2.circle(img, center, radius, color, thickness)
        return
    
    # 计算每段对应的角度（弧度）
    angle_per_segment = 2 * math.pi / num_segments
    dash_angle = (dash_length / circumference) * 2 * math.pi
    
    # 分段绘制
    for i in range(num_segments):
        start_angle = i * angle_per_segment
        end_angle = start_angle + dash_angle
        
        # 计算起点和终点坐标
        x1 = int(x + radius * math.cos(start_angle))
        y1 = int(y + radius * math.sin(start_angle))
        x2 = int(x + radius * math.cos(end_angle))
        y2 = int(y + radius * math.sin(end_angle))
        
        # 绘制小段圆弧（用直线近似）
        # 为了更平滑，可以在每段内再细分
        num_points = max(3, int(dash_angle * radius / 2))
        for j in range(num_points):
            t = j / num_points
            angle = start_angle + t * dash_angle
            px = int(x + radius * math.cos(angle))
            py = int(y + radius * math.sin(angle))
            if j == 0:
                prev_px, prev_py = px, py
            else:
                cv2.line(img, (prev_px, prev_py), (px, py), color, thickness)
                prev_px, prev_py = px, py
                pass
    return 

def Find_Powder_Circle(image_color):
    black_circles = []  # 初始化，避免作用域问题
    image=cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    img_height,img_width = image.shape
    # 二值化  
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)   #TODO 阈值方法，后续调整
    cv2.imshow('binary', binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print(f"图像尺寸: {img_height} x {img_width}")
    print(f"检测参数: minDist={int(img_height*0.5/2)}, minRadius={int(img_height*0.5/2)}, maxRadius={int(img_height/2)}, param2=30")

    circles = cv2.HoughCircles(binary,
        cv2.HOUGH_GRADIENT,
        dp=1,              # 累加器分辨率与图像分辨率的反比
        minDist=int(img_height*0.5/2),        # 检测到的圆心之间的最小距离  {后续改为图片尺寸的1/2}
        param1=100,         # Canny边缘检测的高阈值
        #param2=30,         # 累加器阈值，越小检测到的圆越多（如果未检测到，尝试降低到10-20）
        param2=5,
        minRadius=int(img_height*0.5/2),      # 最小圆半径  {后续改为图片尺寸的1/2}
        maxRadius=int(img_height/2)        # 最大圆半径，0表示不限制 {后续改为图片尺寸}
    )

    # [显示所有找到的圆+筛选出粉桶内圆（蓝色）]
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) # 将灰度图转换为BGR格式以便绘制彩色圆
    black_pixel_count = np.sum(binary == 0)   # 图片二值化后，黑色区域像素数量

    if circles is not None:  # 如果检测到圆，筛选出属于粉桶的圆
        circles = np.around(circles).astype("int")
        print(f"检测到 {len(circles[0])} 个圆")
        
        # 筛选出圆内所有像素都是黑色的圆
        for (x, y, r) in circles[0]:
            # 创建掩码，检查圆内像素
            mask = np.zeros(binary.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)  # 填充圆内区域为255
            
            # 获取圆内的像素值
            circle_pixels = binary[mask == 255]
            
            # 统计黑色像素比例（用于调试）
            black_ratio = np.sum(circle_pixels == 0)/black_pixel_count if len(circle_pixels) > 0 else 0
            print(f"✗ 圆心坐标: ({x}, {y}), 半径: {r} - 圆内黑色像素比例: {black_ratio:.2%}")
            if black_ratio > 0.9:
                black_circles.append((x, y, r))
        print(f"\n筛选结果: 共 {len(black_circles)} 个圆")
        
        
        # 绘制所有检测到的圆_虚线（绿色）
        for (x, y, r) in circles[0]:
            # 对所有找到的圆画虚线
            draw_dashed_circle(binary_color, (x, y), r, (0, 255, 0), 1)

        if black_circles is not None:
            # 输出筛选后的圆的详细信息
            for (x, y, r) in black_circles:
                diameter = 2 * r
                print(f"圆心坐标: ({x}, {y}), 半径: {r}, 直径: {diameter}")
                # 对筛选出真实粉末圆_实线(蓝色)
                cv2.circle(binary_color, (x, y), 3, (0, 0, 255), -1)
                cv2.circle(binary_color, (x, y), r, (0, 0, 255), 2)
        else:
            print("未检测到圆") # [/]

    # [仅显示判为粉末的圆（蓝色）]
    if black_circles is not None:
        cv2.imshow('binary with all circles (green)', binary_color)
        cv2.waitKey(0)
        
        # 显示筛选后的圆（蓝色 - 圆内全为黑色）
        binary_color_filtered = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) # 为了画出蓝色
        for (x, y, r) in black_circles:
            # 绘制圆心（蓝色）
            cv2.circle(binary_color_filtered, (x, y), 3, (255, 0, 0), -1)
            # 绘制圆（蓝色）
            cv2.circle(binary_color_filtered, (x, y), r, (255, 0, 0), 2)
        cv2.imshow('Powder Circle (blue)', binary_color_filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        # [/]
    return black_circles

def are_point_in_circle(points, center, r):
    """
    判断点坐标是否在以(x,y)为圆心，r为半径的圆范围内
    
    参数:
        point: 点坐标 (px, py) 或 numpy数组
        center: 圆心坐标 (x, y)
        r: 圆的半径
    
    返回:
        如果点在圆内（包括圆上）返回point，否则返回 None
    """
    points_rslt=[]
    for point in points:  
        px, py = point  # BUG 报错
        x, y = center
        distance = math.sqrt((px - x)**2 + (py - y)**2)   
        if distance <= r:
            points_rslt.append(point)
        else:
            pass
    return points_rslt
    
# 线不用
def create_circular_mask(image, r, center=None):
    """
    制作圆形半径为r的mask，mask区域内像素值保留，其余变为零
    
    参数:
        image: 输入图像（可以是灰度图或彩色图）
        r: 圆形mask的半径
        center: 圆心坐标 (x, y)，如果为None则使用图像中心
    
    返回:
        处理后的图像，mask区域外像素值为0
    """
    # 获取图像尺寸
    if len(image.shape) == 2:
        # 灰度图
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        # 彩色图
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
    
    # 如果没有指定圆心，使用图像中心
    if center is None:
        center = (w // 2, h // 2)
    
    # 创建圆形mask（填充圆内区域为255）
    cv2.circle(mask, center, r, 255, -1)
    
    # 应用mask：mask区域内保留原值，其余变为0
    if len(image.shape) == 2:
        # 灰度图
        result = cv2.bitwise_and(image, image, mask=mask)
    else:
        # 彩色图
        result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

def Find_Contours(image):
    max_contour_num=0
    max_contour_idx=0
    inner_contour=[]
    binary=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 获取轮廓  #TODO 后续更换获取轮廓的方式
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 同时获得内外轮廓，外轮廓指图片矩形框
    # 筛选出内轮廓（有父轮廓的轮廓）
    # [hierarchy格式说明]
    # hierarchy格式: [Next, Previous, First_Child, Parent]
    # hierarchy[0][i] 是一个4元素数组：
    #   [0] - Next: 同一层级的下一个轮廓索引
    #   [1] - Previous: 同一层级的上一个轮廓索引
    #   [2] - First_Child: 第一个子轮廓索引
    #   [3] - Parent: 父轮廓索引（如果为-1表示没有父轮廓，即外轮廓）
    # 如果Parent >= 0，说明有父轮廓，是内轮廓
    # [/]
    #inner_contours = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] >= 0] # 原版先保留： 找出轮廓数量最多的轮廓
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] >= 0:
            #inner_contour.append(contour)
            if len(contour) > max_contour_num:
                max_contour_num = len(contour)
                max_contour_idx = i
    print(f"轮廓长度{max_contour_num}")  # inner_contour只有一个，原版inner_contours有很多个
    inner_contour = contours[max_contour_idx]
    # 将灰度图转换为BGR格式以便绘制彩色轮廓
    image_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_color, contours[max_contour_idx], -1, (0, 0, 255), 1) # 显示最长轮廓
    #cv2.drawContours(image_color, inner_contours, -1, (0, 0, 255), 1) # 原版，全部轮廓都画成一样颜色
    # [为每个轮廓绘制不同颜色 ，问题溯源用]
    # num_contours = len(inner_contours)
    # print(f"num of inner_contours: {num_contours}")
    # for i, contour in enumerate(inner_contours):
    #     # 使用HSV颜色空间生成均匀分布的颜色，然后转换为BGR
    #     hue = int(180 * i / max(1, num_contours - 1))  # 0-180的色相值
    #     color_hsv = np.uint8([[[hue, 255, 255]]])
    #     color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    #     color_bgr = tuple(map(int, color_bgr))
    #     # 绘制单个轮廓
    #     cv2.drawContours(image_color, [contour], -1, color_bgr, 1)  
    # [/]
    cv2.imshow('image', image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return inner_contour

if __name__ == '__main__':
    image = cv2.imread('./Pic/h.png', cv2.IMREAD_COLOR)
    black_circles = Find_Powder_Circle(image)
    x,y,r_raw=black_circles[0]
    r=int(0.9*r_raw)
    print(f"圆心坐标: ({x}, {y}), 0.9R 半径: {r}")
    inner_contour = Find_Contours(image)  # 改成只有1条了
    # 消除维度为1的维度，将 (n, 1, 2) 转换为 (n, 2)
    # FIXME 找到contours的数据结构
    #inner_contour = [np.squeeze(contour, axis=1) for contour in inner_contour]
    split_line=[]  # 分割线列表
    split_line=np.squeeze(inner_contour, axis=1)
    split_line=split_line.squeeze()
    split_line=are_point_in_circle(split_line, (x, y), r)
    # 将点改为符合contours的数据结构
    split_line_contour=(np.array(split_line)).reshape(-1, 1, 2)
    cv2.drawContours(image, split_line_contour, -1, (0, 255, 0), 1)
    cv2.imshow('split_line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Completed") 
