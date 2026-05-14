import math

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import laspy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import argparse
import re
import json
import copy

class ImageParameters:
    """图像参数类"""
    def __init__(self, image_name, x, y, z, roll, pitch, yaw, extra_params=None):
        self.image_name = image_name
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.extra_params = extra_params or []

    def __str__(self):
        return (f"图像: {self.image_name}, "
                f"位置: ({self.x:.6f}, {self.y:.6f}, {self.z:.6f}), "
                f"姿态: Roll={np.degrees(self.roll):.3f}°, "
                f"Pitch={np.degrees(self.pitch):.3f}°, "
                f"Yaw={np.degrees(self.yaw):.3f}°")

    @classmethod
    def from_parameter_string(cls, param_string):
        """
        从参数字符串解析图像参数
        
        格式: 图像名称 空值1 空值2 x y z roll pitch yaw [其他参数...]
        示例: "8265.918148_IMG.jpeg 0.000000000 0.000000000 0.039 -0.048 0.062 -1.311141 -0.032633 1.576580 8265.9181"
        
        Args:
            param_string: 参数字符串
            
        Returns:
            ImageParameters对象
        """
        parts = param_string.strip().split()

        if len(parts) < 9:
            raise ValueError(f"参数字符串格式错误，至少需要9个参数，实际得到{len(parts)}个")

        # 解析各个参数
        image_name = parts[0]
        # parts[1] 和 parts[2] 是空值，跳过
        x = float(parts[3])
        y = float(parts[4])
        z = float(parts[5])
        roll = float(parts[6])  # 弧度
        pitch = float(parts[7])  # 弧度
        yaw = float(parts[8])  # 弧度

        # 其他额外参数
        extra_params = [float(p) for p in parts[9:]] if len(parts) > 9 else []

        return cls(image_name, x, y, z, roll, pitch, yaw, extra_params)


class PanoramicProjector:
    def __init__(self, image_params, image_base_path=""):
        """
        初始化全景投影器
        
        Args:
            image_params: ImageParameters对象
            image_base_path: 图像文件的基础路径
        """
        # 构建完整的图像路径
        image_path = os.path.join(image_base_path, image_params.image_name)

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        self.img_h, self.img_w = self.image.shape[:2]
        self.image_params = image_params
        self.camera_pos = np.array([image_params.x, image_params.y, image_params.z])

        # 计算旋转矩阵
        self.rotation_matrix = self.get_rotation_from_angle(
            image_params.roll, image_params.pitch, image_params.yaw
        )

        print(f"图像: {image_params.image_name}")
        print(f"图像尺寸: {self.img_w} x {self.img_h}")
        print(f"相机位置: ({image_params.x:.6f}, {image_params.y:.6f}, {image_params.z:.6f})")
        print(f"姿态角 (弧度): Roll={image_params.roll:.6f}, Pitch={image_params.pitch:.6f}, Yaw={image_params.yaw:.6f}")
        print(f"姿态角 (度): Roll={np.degrees(image_params.roll):.3f}°, Pitch={np.degrees(image_params.pitch):.3f}°, Yaw={np.degrees(image_params.yaw):.3f}°")
        if image_params.extra_params:
            print(f"额外参数: {image_params.extra_params}")

    def get_rotation_from_angle(self, in_roll, in_pitch, in_yaw):
        """
        将角度转换为旋转矩阵，参考C++代码实现
        
        Args:
            in_roll, in_pitch, in_yaw: 姿态角（弧度）
            
        Returns:
            3x3旋转矩阵
        """
        roll = -in_roll
        pitch = -in_pitch
        yaw = in_yaw

        R = np.zeros((3, 3))

        R[0, 0] = np.cos(roll) * np.cos(yaw) + np.sin(pitch) * np.sin(roll) * np.sin(yaw)
        R[0, 1] = -np.cos(roll) * np.sin(yaw) + np.sin(pitch) * np.sin(roll) * np.cos(yaw)
        R[0, 2] = np.cos(pitch) * np.sin(roll)

        R[1, 0] = np.cos(pitch) * np.sin(yaw)
        R[1, 1] = np.cos(pitch) * np.cos(yaw)
        R[1, 2] = -np.sin(pitch)

        R[2, 0] = -np.sin(roll) * np.cos(yaw) + np.sin(pitch) * np.cos(roll) * np.sin(yaw)
        R[2, 1] = np.sin(roll) * np.sin(yaw) + np.sin(pitch) * np.cos(roll) * np.cos(yaw)
        R[2, 2] = np.cos(roll) * np.cos(pitch)

        return R

    def coordinate_to_pixel(self, pt):
        """
        将全景坐标系下的3D点转换为全景图像像素坐标
        
        Args:
            pt: 3D点坐标 [x, y, z]
            
        Returns:
            [u, v] 像素坐标
        """
        # 水平角度计算
        tm = np.arctan(-pt[0] / pt[1]) if pt[1] != 0 else 0

        if pt[0] < 0:
            if pt[1] > 0:
                u = int(self.img_h - self.img_h * tm / np.pi + 0.5) % self.img_w
            else:
                u = int(self.img_h * (-tm) / np.pi + 0.5) % self.img_w
        else:
            if pt[1] > 0:
                u = int(self.img_h - self.img_h * tm / np.pi + 0.5) % self.img_w
            else:
                u = int(self.img_w - self.img_h * tm / np.pi + 0.5) % self.img_w

        # 垂直角度计算
        horizontal_dist = np.sqrt(pt[0] * pt[0] + pt[1] * pt[1])
        tm = np.arctan(horizontal_dist / pt[2]) if pt[2] != 0 else np.pi/2

        if tm < 0:
            tm = tm + np.pi

        v = int(self.img_h * tm / np.pi + 0.5) % self.img_h

        return [u, v]

    def calculate_distance_to_camera(self, point):
        """
        计算点到相机的欧几里得距离
        
        Args:
            point: 3D点坐标 [x, y, z]
            
        Returns:
            float: 距离值
        """
        return np.linalg.norm(point - self.camera_pos)

    def set_pixel_color(self, image, x, y, color_bgr):
        """
        直接设置像素的BGR颜色值
        
        Args:
            image: 图像数组
            x, y: 像素坐标
            color_bgr: BGR颜色元组 (B, G, R)
        """
        # 确保坐标在图像范围内
        if 0 <= x < self.img_w and 0 <= y < self.img_h:
            # OpenCV使用BGR格式，直接设置像素值
            image[y, x] = color_bgr

    def set_pixel_area(self, image, center_x, center_y, color_bgr, point_size):
        """
        设置以(center_x, center_y)为中心的方形或圆形区域的像素颜色
        
        Args:
            image: 图像数组
            center_x, center_y: 中心坐标
            color_bgr: BGR颜色元组
            point_size: 点大小（半径）
        """
        if point_size <= 1:
            # 只设置单个像素
            self.set_pixel_color(image, center_x, center_y, color_bgr)
        else:
            # 设置方形区域的所有像素
            for dy in range(-point_size, point_size + 1):
                for dx in range(-point_size, point_size + 1):
                    # 可选：使用圆形区域而非方形
                    # if dx*dx + dy*dy <= point_size*point_size:
                    px, py = center_x + dx, center_y + dy
                    self.set_pixel_color(image, px, py, color_bgr)

    def read_las_pointcloud(self, las_path):
        """
        读取LAS点云文件
        
        Args:
            las_path: LAS文件路径
            
        Returns:
            points: 点云坐标数组 (N, 3)
            intensities: 强度数组 (N,)
        """
        print(f"正在读取点云文件: {las_path}")

        with laspy.open(las_path) as las_file:
            las = las_file.read()

            points = np.vstack([las.x, las.y, las.z]).T

            # 获取强度信息
            if hasattr(las, 'intensity'):
                intensities = las.intensity
            else:
                print("警告: 点云文件中没有强度信息，使用默认值")
                intensities = np.ones(len(points)) * 128

            #获取颜色值
            colors = None
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                # LAS文件中的颜色值通常是16位(0-65535)，需要转换为8位(0-255)
                colors = np.vstack([
                    las.red / 65535.0 * 255,
                    las.green / 65535.0 * 255,
                    las.blue / 65535.0 * 255
                ]).T.astype(np.uint8)
                print(f"读取到颜色信息")
            else:
                print("警告: 点云文件中没有颜色信息")

        print(f"读取到 {len(points)} 个点")
        return points, intensities, colors

    def project_pointcloud_with_depth(self, points, intensities, output_path=None, point_size=1, camera_pos=None, colors = None):
        """
        将点云投影到全景图像上，使用深度判定确保近距离点覆盖远距离点
        
        Args:
            points: 点云坐标数组 (N, 3)
            intensities: 强度数组 (N,)
            output_path: 输出图像路径，默认使用图像名称生成
            point_size: 投影点的大小（像素半径）
            
        Returns:
            result_image: 投影结果图像
            projected_count: 成功投影的点数量
            depth_stats: 深度统计信息
        """
        # 如果没有指定输出路径，使用图像名称生成
        if output_path is None:
            base_name = os.path.splitext(self.image_params.image_name)[0]
            output_path = f"{base_name}_projected_depth.jpg"

        # 创建输出图像副本和深度缓冲区
        result_image = self.image.copy()
        depth_buffer = np.full((self.img_h, self.img_w), np.inf, dtype=np.float32)

        # 强度归一化，用于颜色映射
        norm = Normalize(vmin=intensities.min(), vmax=intensities.max())
        colormap = plt.cm.jet  # 使用jet色彩映射

        projected_count = 0
        depth_updated_count = 0
        valid_depths = []

        print("开始投影点云（使用深度判定）...")
        print(f"点大小: {point_size} 像素")

        # 预计算所有点的距离，用于排序和统计
        distances = np.array([self.calculate_distance_to_camera(point) for point in points])
        valid_mask = distances > 0  # 过滤无效距离

        print(f"距离统计: 最小={distances[valid_mask].min():.3f}m, "
              f"最大={distances[valid_mask].max():.3f}m, "
              f"平均={distances[valid_mask].mean():.3f}m")

        for i, (point, intensity, distance) in enumerate(zip(points, intensities, distances)):
            if i % 10000 == 0:
                print(f"\r处理进度: {i}/{len(points)} ({i/len(points)*100:.1f}%)")

            # 跳过无效距离的点
            if distance <= 0:
                continue

            # 点云坐标减去相机位置
            pt1 = point - self.camera_pos

            # 应用旋转矩阵变换
            pt2 = self.rotation_matrix @ pt1

            # 转换为像素坐标
            try:
                uv = self.coordinate_to_pixel(pt2)
                u, v = uv[0], uv[1]

                # 检查像素坐标是否在图像范围内
                if 0 <= u < self.img_w and 0 <= v < self.img_h:
                    projected_count += 1

                    # 根据强度值获取颜色
                    color_norm = norm(intensity)
                    color_rgba = colormap(color_norm)
                    # OpenCV使用BGR格式
                    color_bgr = (int(color_rgba[2] * 255),
                               int(color_rgba[1] * 255),
                               int(color_rgba[0] * 255))

                    # 深度判定：只有当前点距离更近时才更新像素
                    pixel_updated = False

                    if point_size <= 1:
                        # 单像素处理
                        if distance < depth_buffer[v, u]:
                            depth_buffer[v, u] = distance
                            target_color = result_image[v,u]
                            source_color = np.asarray([colors[i][2], colors[i][1], colors[i][0]])
                            f = np.all(np.abs(source_color - target_color) < 10)
                            if f:
                                print("111")
                                result_image[v, u] = color_bgr  # 直接设置像素值
                            # else:
                            #     print(f"{target_color}过滤")

                            pixel_updated = True
                    else:
                        # 多像素区域处理
                        for dy in range(-point_size, point_size + 1):
                            for dx in range(-point_size, point_size + 1):
                                px, py = u + dx, v + dy

                                # 确保像素在图像范围内
                                if 0 <= px < self.img_w and 0 <= py < self.img_h:
                                    # 深度测试
                                    if distance < depth_buffer[py, px]:
                                        depth_buffer[py, px] = distance
                                        result_image[py, px] = color_bgr  # 直接设置像素值
                                        pixel_updated = True

                    if pixel_updated:
                        depth_updated_count += 1
                        valid_depths.append(distance)

            except Exception as e:
                # 跳过投影失败的点
                continue

        # 计算深度统计
        if valid_depths:
            depth_stats = {
                'min_depth': min(valid_depths),
                'max_depth': max(valid_depths),
                'mean_depth': np.mean(valid_depths),
                'median_depth': np.median(valid_depths),
                'std_depth': np.std(valid_depths)
            }
        else:
            depth_stats = {}

        print(f"投影完成!")
        print(f"- 总点数: {len(points)}")
        print(f"- 投影到图像内的点数: {projected_count}")
        print(f"- 实际渲染的点数: {depth_updated_count}")
        print(f"- 投影率: {projected_count/len(points)*100:.2f}%")
        print(f"- 渲染率: {depth_updated_count/len(points)*100:.2f}%")

        # 保存结果图像
        cv2.imwrite(output_path, result_image)
        print(f"结果已保存到: {output_path}")

        # 可选：保存深度图
        depth_image_path = output_path.replace('.jpg', '_depth.png')
        self.save_depth_image(depth_buffer, depth_image_path)

        return result_image, projected_count, depth_stats

    def pixel_to_sphere(self, u, v, height, width):
        # longitude = (2 * np.pi * u / width) - np.pi  # 经度范围: [-π, π] fi
        # latitude = (np.pi * v / height) - (np.pi / 2)  # 纬度范围: [-π/2, π/2] theta
        #
        # # 转换为笛卡尔坐标
        # x = 16 * np.sin(latitude) * np.cos(longitude)
        # y = 16 * np.sin(latitude) * np.sin(longitude)
        # z = 16 * np.cos(latitude)
        #
        # r = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.float32)
        # xyz = r @ np.array([x, y, z]).T
        #
        # return [xyz[0], xyz[1], -xyz[2]]



        radius = 16
        #
        # v = -v + height
        #
        # phi = u / width * (2 * np.pi)
        # theta = v / height * np.pi
        #
        # x = -radius * math.cos(phi) * math.sin(theta)
        # y = radius * math.cos(theta)
        # z = radius * math.sin(phi) * math.sin(theta)

        # return [x, y, -z]
        pass

    def project_camera_pos(self, camera_pos, output_path):
        img_height = 5632
        img_width = 11264
        # 提取所有图像位置坐标
        import matplotlib.pyplot as plt

        move_map = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        # for i in range(img_height):
        #     for j in range(img_width):
        #         move_map[i][j] = [0, 0, 255]
        # cv2.imwrite( "move_.jpg", move_map)
        # cv2.imshow("Red Image", move_map)
        # cv2.waitKey(0)  # 等待按键输入
        # cv2.destroyAllWindows()
        # return
        image_pos_list = []
        t = 0
        for pos_info in camera_pos:
            image_pos = dict()
            point = np.array([pos_info["image_pos"]["x"], pos_info["image_pos"]["y"], pos_info["image_pos"]["z"]])
            dist = self.calculate_distance_to_camera(point)
            if dist <= 0:
                continue
            pt1 = point - self.camera_pos
            pt2 = self.rotation_matrix @ pt1
            uv = self.coordinate_to_pixel(pt2)
            u, v = uv[0], uv[1]

            # 检查像素坐标是否在图像范围内
            if 0 <= u < img_width and 0 <= v < img_height:
                image_pos["image_name"] = pos_info["image_name"]
                image_pos["height"] = img_height
                image_pos["width"] = img_width
                image_pos["uv"] = [u, v]
                image_pos["sphere_pos"] = self.pixel_to_sphere(u, v, img_height, img_width)

                image_pos["image_pos"] = pos_info["image_pos"]
                image_pos_list.append(image_pos)

                t += 1
                # 绘制半径为10的红色圆形区域
                radius = 10
                # 计算圆形区域的边界范围（减少循环次数）
                min_u = max(0, u - radius)
                max_u = min(img_width - 1, u + radius)
                min_v = max(0, v - radius)
                max_v = min(img_height - 1, v + radius)

                # 遍历圆形区域内的所有像素
                for current_v in range(min_v, max_v + 1):
                    for current_u in range(min_u, max_u + 1):
                        # 计算与中心点的距离平方
                        dx = current_u - u
                        dy = current_v - v
                        if dx * dx + dy * dy <= radius * radius:
                            # 设置为红色 [255, 0, 0]
                            print(f"{u},{v}")
                            move_map[current_v][current_u] = np.array([255, 0, 0])
        print(t)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(image_pos_list, indent=4, ensure_ascii=False))
        plt.imsave("move_map.jpg", move_map)
        # print(move_map.dtype)
        # print(move_map.shape)
        #cv2.imwrite("move_map.jpg", move_map)
        # cv2.imshow("Red Image", move_map)
        # cv2.waitKey(0)  # 等待按键输入
        # cv2.destroyAllWindows()
    def save_depth_image(self, depth_buffer, output_path):
        """
        保存深度图像
        
        Args:
            depth_buffer: 深度缓冲区
            output_path: 输出路径
        """
        # 创建深度可视化图像
        valid_depths = depth_buffer[depth_buffer != np.inf]
        if len(valid_depths) > 0:
            # 归一化深度值到0-255
            min_depth = valid_depths.min()
            max_depth = valid_depths.max()

            depth_normalized = np.zeros_like(depth_buffer, dtype=np.uint8)
            valid_mask = depth_buffer != np.inf
            depth_normalized[valid_mask] = ((depth_buffer[valid_mask] - min_depth) /
                                          (max_depth - min_depth) * 255).astype(np.uint8)

            # 应用色彩映射
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # 将无效像素设为黑色
            depth_colored[~valid_mask] = [0, 0, 0]

            cv2.imwrite(output_path, depth_colored)
            print(f"深度图已保存到: {output_path}")

    # 保留原有的投影方法作为备选
    def project_pointcloud(self, points, intensities, output_path=None):
        """
        将点云投影到全景图像上（原始方法，无深度判定）
        """
        return self.project_pointcloud_with_depth(points, intensities, output_path, point_size=1)

    def create_intensity_colorbar(self, intensities, output_path=None):
        """
        创建强度色彩条
        
        Args:
            intensities: 强度数组
            output_path: 色彩条输出路径，默认使用图像名称生成
        """
        if output_path is None:
            base_name = os.path.splitext(self.image_params.image_name)[0]
            output_path = f"{base_name}_colorbar.png"

        fig, ax = plt.subplots(figsize=(8, 1))
        norm = Normalize(vmin=intensities.min(), vmax=intensities.max())

        # 创建色彩条
        colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'),
                               cax=ax, orientation='horizontal')
        colorbar.set_label('点云强度')

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"色彩条已保存到: {output_path}")


def parse_parameter_file(file_path):
    """
    解析包含多个图像参数的文件
    
    Args:
        file_path: 参数文件路径
        
    Returns:
        list: ImageParameters对象列表
    """
    parameters = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过空行和注释行
                continue

            try:
                params = ImageParameters.from_parameter_string(line)
                parameters.append(params)
            except Exception as e:
                print(f"警告: 第{line_num}行参数解析失败: {e}")
                continue

    return parameters

def get_camera_pos(camera_pos_path):
    with open(camera_pos_path, 'r', encoding='utf-8') as file:
        camera_pos = json.load(file)
        print("camera_pos load finish!\n")
    return camera_pos

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='点云全景投影工具(直接像素操作)')
    parser.add_argument('--params', type=str, help='图像参数字符串 (格式: "图像名 0 0 x y z roll pitch yaw")')
    parser.add_argument('--param_file', type=str, help='包含多个图像参数的文件路径')
    parser.add_argument('--pointcloud', type=str, required=True, help='LAS点云文件路径')
    parser.add_argument('--image_path', type=str, default="", help='图像文件基础路径')
    parser.add_argument('--output', type=str, help='输出图像路径 (可选，默认使用图像名称生成)')
    parser.add_argument('--point_size', type=int, default=1, help='投影点的大小（像素半径），默认为1')
    parser.add_argument('--no_depth', action='store_true', help='禁用深度判定（使用原始投影方法）')
    parser.add_argument('--camera_pos', type=str, help='图像位置')


    args = parser.parse_args()

    # 验证参数
    if not args.params and not args.param_file:
        print("错误: 必须提供 --params 或 --param_file 参数")
        return 1

    if args.params and args.param_file:
        print("错误: --params 和 --param_file 不能同时使用")
        return 1

    try:
        # 解析图像参数
        if args.params:
            # 单个参数字符串
            image_params = ImageParameters.from_parameter_string(args.params)
            parameters_list = [image_params]
        else:
            # 从文件读取多个参数
            parameters_list = parse_parameter_file(args.param_file)
            if not parameters_list:
                print("错误: 没有成功解析到任何图像参数")
                return 1
        #
        # args.camera_pos = "camera_pos.json"
        # camera_pos = get_camera_pos(args.camera_pos)
        #
        # #temp
        # projector_temp = PanoramicProjector(parameters_list[0], args.image_path)
        # projector_temp.project_camera_pos(camera_pos, "camera_pos_info.json")
        # return

        # 处理每个图像参数
        for i, image_params in enumerate(parameters_list):
            print(f"\n{'='*60}")
            print(f"处理第 {i+1}/{len(parameters_list)} 个图像")
            print(f"{'='*60}")
            print(image_params)

            # 创建投影器
            projector = PanoramicProjector(image_params, args.image_path)

            # 读取点云
            points, intensities, colors = projector.read_las_pointcloud(args.pointcloud)

            # 设置输出路径
            output_path = args.output if len(parameters_list) == 1 else None

            # 执行投影
            if args.no_depth:
                # 使用原始投影方法
                result_image, projected_count = projector.project_pointcloud(
                    points, intensities, output_path
                )
                depth_stats = {}
            else:
                # 使用深度判定投影
                result_image, projected_count, depth_stats = projector.project_pointcloud_with_depth(
                    points, intensities, output_path, args.point_size, colors = colors
                )

            # 创建色彩条
            projector.create_intensity_colorbar(intensities)

            print(f"\n最终结果:")
            print(f"- 输入点云: {len(points)} 个点")
            print(f"- 成功投影: {projected_count} 个点")
            print(f"- 投影率: {projected_count/len(points)*100:.2f}%")

    except Exception as e:
        print(f"错误: {str(e)}")
        return 1

    return 0


def example_usage():
    """
    示例用法
    """
    # 示例参数字符串
    param_string = "8265.918148_IMG.jpeg 0.000000000 0.000000000 0.039 -0.048 0.062 -1.311141 -0.032633 1.576580 8265.9181"

    print(f"输入字符串: {param_string}")

    print("\n命令行用法:")
    print(f'python pointcloud_projection.py \\')
    print(f'    --params "{param_string}" \\')
    print(f'    --pointcloud pointcloud.las \\')
    print(f'    --image_path ./images/ \\')
    print(f'    --point_size 2 \\')
    print(f'    --output result.jpg')

    print("\n编程用法:")
    print(f"""
        # 从字符串解析参数
        params = ImageParameters.from_parameter_string(
            "{param_string}"
        )

        # 创建投影器
        projector = PanoramicProjector(params, image_base_path="./images/")

        # 读取点云
        points, intensities = projector.read_las_pointcloud("pointcloud.las")

        # 使用深度判定投影（直接像素操作）
        result_image, count, depth_stats = projector.project_pointcloud_with_depth(
            points, intensities, point_size=2
        )

        # 创建色彩条
        projector.create_intensity_colorbar(intensities)
        """)



if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        example_usage()
    else:
        main()
