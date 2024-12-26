import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# 基本目录和场景定义
BASE_DIR = "FruitImages"
SCENARIOS = ["ScenarioA", "ScenarioB", "ScenarioC", "ScenarioD"]  # ScenarioD为选做


class FruitDetectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("水果识别与计数 (精确版)")
        self.master.geometry("2000x1000")  # 增大窗口大小以适应更大的输出图像

        # ===================== GUI布局：顶部场景选择和图片列表 =====================
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # 场景选择下拉框
        tk.Label(top_frame, text="选择场景:").pack(side=tk.LEFT, padx=5)
        self.scenario_var = tk.StringVar()
        self.combo_scenario = ttk.Combobox(
            top_frame, textvariable=self.scenario_var, values=SCENARIOS, width=15, state="readonly"
        )
        self.combo_scenario.pack(side=tk.LEFT, padx=5)
        self.combo_scenario.current(0)  # 默认选中第一个
        self.combo_scenario.bind("<<ComboboxSelected>>", self.on_scenario_changed)

        # 图片列表框
        tk.Label(top_frame, text="选择图片:").pack(side=tk.LEFT, padx=5)
        self.listbox_images = tk.Listbox(top_frame, width=30, height=10)
        self.listbox_images.pack(side=tk.LEFT, padx=5)
        self.listbox_images.bind("<<ListboxSelect>>", self.on_image_selected)

        # 初始化Listbox内容
        self.update_image_listbox(SCENARIOS[0])

        # ===================== 中央：图片显示区 =====================
        # 使用grid布局将处理阶段的Canvas和输出Canvas分开
        center_frame = tk.Frame(self.master)
        center_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 设置center_frame的行和列权重
        center_frame.grid_columnconfigure(0, weight=1)
        center_frame.grid_columnconfigure(1, weight=2)  # 右侧输出Canvas占更多空间
        center_frame.grid_rowconfigure(0, weight=1)

        # 左侧：处理阶段的Canvas
        processing_frame = tk.Frame(center_frame)
        processing_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 设置processing_frame的行和列权重
        for i in range(3):  # 3行
            processing_frame.grid_rowconfigure(i, weight=1)
        for j in range(2):  # 2列
            processing_frame.grid_columnconfigure(j, weight=1)

        # 创建处理阶段的Canvas并 grid 位置
        self.canvas_original = self.create_canvas(processing_frame, "原始图像", 0, 0)
        self.canvas_hsv = self.create_canvas(processing_frame, "HSV转换图像", 0, 1)
        self.canvas_masks = self.create_canvas(processing_frame, "颜色掩膜", 1, 0)
        self.canvas_morph = self.create_canvas(processing_frame, "形态学处理后的掩膜", 1, 1)
        self.canvas_markers = self.create_canvas(processing_frame, "分水岭标记图", 2, 0)
        self.canvas_gradient = self.create_canvas(processing_frame, "梯度图像", 2, 1)  # 新增

        # 右侧：输出Canvas
        output_frame = tk.Frame(center_frame)
        output_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        tk.Label(output_frame, text="识别结果图像").pack()
        self.canvas_output = tk.Canvas(output_frame, bg="#ECECEC")
        self.canvas_output.pack(fill=tk.BOTH, expand=True)

        # ===================== 底部：按钮和识别结果显示 =====================
        bottom_frame = tk.Frame(self.master)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.btn_detect = tk.Button(
            bottom_frame, text="识别水果", command=self.detect_fruit, width=15
        )
        self.btn_detect.pack(side=tk.LEFT, padx=10)

        self.label_result = tk.Label(
            bottom_frame, text="识别结果将在此处显示", bg="#F0F8FF", anchor='nw', justify='left', padx=10, pady=10
        )
        self.label_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        # ===================== 数据变量 =====================
        self.image_path = None
        self.cv_image = None  # OpenCV格式的图像
        self.display_image = None  # PIL格式用于显示
        self.pil_images = {}  # 存储各阶段的PIL图像
        self.tk_images = {}  # 存储各阶段的ImageTk图像

        # 地面真实数据
        self.ground_truth = {}
        self.load_ground_truth("ground_truth.txt")  # 加载地面真实数据

        # 绑定窗口大小变化事件，以动态调整图片大小
        self.master.bind("<Configure>", self.on_resize)

    # 创建带标签的Canvas，使用grid布局
    def create_canvas(self, parent, label_text, row, column):
        frame = tk.Frame(parent)
        frame.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)
        tk.Label(frame, text=label_text).grid(row=0, column=0, padx=5, pady=5)
        canvas = tk.Canvas(frame, bg="#ECECEC")
        canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        return canvas

    # -----------------------------------------------------------------
    # 根据选择的场景更新Listbox中的图片文件列表
    def update_image_listbox(self, scenario):
        self.listbox_images.delete(0, tk.END)
        scenario_dir = os.path.join(BASE_DIR, scenario)
        if not os.path.isdir(scenario_dir):
            messagebox.showerror("错误", f"目录 {scenario_dir} 不存在！")
            return

        # 获取有效的图片文件
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        files = [
            f for f in os.listdir(scenario_dir)
            if f.lower().endswith(valid_ext)
        ]
        files.sort()  # 排序，便于用户查找

        for f in files:
            self.listbox_images.insert(tk.END, f)

    def on_scenario_changed(self, event=None):
        scenario = self.scenario_var.get()
        self.update_image_listbox(scenario)

    def on_image_selected(self, event=None):
        # 当Listbox选中后，自动加载图片
        selection = self.listbox_images.curselection()
        if not selection:
            return
        idx = selection[0]
        filename = self.listbox_images.get(idx)
        scenario = self.scenario_var.get()
        self.image_path = os.path.join(BASE_DIR, scenario, filename)

        self.load_and_show_image(self.image_path)

    # -----------------------------------------------------------------
    # 加载地面真实数据
    def load_ground_truth(self, filepath):
        """
        读取地面真实数据文件，并存储在self.ground_truth字典中。
        字典结构：{ 'ScenarioA/A1.jpg': {'Apple': 10, 'Banana': 5, 'Orange': 15}, ... }
        """
        if not os.path.isfile(filepath):
            messagebox.showerror("错误", f"地面真实数据文件 {filepath} 不存在！")
            return

        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                parts = line.split(',')
                image_path = parts[0].strip()
                fruit_counts = {}
                for part in parts[1:]:
                    try:
                        fruit, count = part.strip().split(':')
                        fruit_counts[fruit] = int(count)
                    except ValueError:
                        messagebox.showwarning("警告", f"地面真实数据文件中存在格式错误的行：{line}")
                        continue
                self.ground_truth[image_path] = fruit_counts

    # -----------------------------------------------------------------
    # 统一：加载并在GUI显示图像
    def load_and_show_image(self, path):
        try:
            # 用 PIL 打开
            pil_img = Image.open(path)

            # 用 OpenCV 读取做后续处理
            self.cv_image = cv2.imread(path)

            # 计算每个Canvas的大小
            canvas_width = self.canvas_original.winfo_width()
            canvas_height = self.canvas_original.winfo_height()

            if canvas_width < 10 or canvas_height < 10:
                # 如果窗口尚未完全渲染，使用主窗口的大小减去预留空间
                canvas_width = self.master.winfo_width() // 4
                canvas_height = self.master.winfo_height() // 3

            # 自适应大小：保持纵横比
            pil_img_resized = self.resize_image_keep_ratio(pil_img, canvas_width, canvas_height)

            self.display_image = pil_img_resized  # 保存PIL格式的图像
            self.pil_images["original"] = pil_img_resized

            # 转成 PhotoImage
            self.tk_images["original"] = ImageTk.PhotoImage(pil_img_resized)

            # 清空Canvas并显示新图片
            self.canvas_original.delete("all")
            self.canvas_original.create_image(
                canvas_width / 2, canvas_height / 2,
                image=self.tk_images["original"], anchor=tk.CENTER
            )

            # 清空其他Canvas
            self.clear_other_canvases()

            self.label_result.config(text=f"图片 [{os.path.basename(path)}] 加载完毕，请点击『识别水果』开始分析。")
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片：{e}")

    # 清空其他Canvas的图像
    def clear_other_canvases(self):
        for key in ["hsv", "masks", "morph", "markers", "gradient", "output"]:
            if key in self.tk_images:
                del self.tk_images[key]
            if key in self.pil_images:
                del self.pil_images[key]
        self.canvas_hsv.delete("all")
        self.canvas_masks.delete("all")
        self.canvas_morph.delete("all")
        self.canvas_markers.delete("all")
        self.canvas_gradient.delete("all")  # 清空新增的Canvas
        self.canvas_output.delete("all")

    @staticmethod
    def resize_image_keep_ratio(pil_img, max_w, max_h):
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h)
        scale = min(scale, 1)  # 不放大图片，只缩小
        new_w, new_h = int(w * scale), int(h * scale)
        try:
            return pil_img.resize((new_w, new_h), resample=Image.LANCZOS)
        except AttributeError:
            # Pillow版本较旧，使用ANTIALIAS代替
            return pil_img.resize((new_w, new_h), resample=Image.ANTIALIAS)

    def on_resize(self, event):
        """窗口大小变化时，重新调整显示的图片大小"""
        if self.display_image is None:
            return

        # 重新调整原始图片大小
        canvas_width = self.canvas_original.winfo_width()
        canvas_height = self.canvas_original.winfo_height()
        pil_img_resized = self.resize_image_keep_ratio(self.display_image, canvas_width, canvas_height)
        self.pil_images["original"] = pil_img_resized
        self.tk_images["original"] = ImageTk.PhotoImage(pil_img_resized)

        # 清空Canvas并显示调整后的图片
        self.canvas_original.delete("all")
        self.canvas_original.create_image(
            canvas_width / 2, canvas_height / 2,
            image=self.tk_images["original"], anchor=tk.CENTER
        )

        # 同时调整其他显示的图像（如果有的话）
        for key in ["hsv", "masks", "morph", "markers", "gradient", "output"]:
            if key in self.pil_images:
                target_canvas = getattr(self, f'canvas_{key}')
                canvas_w = target_canvas.winfo_width()
                canvas_h = target_canvas.winfo_height()

                # 重新调整图像大小
                pil_img = self.pil_images[key]
                pil_img_resized = self.resize_image_keep_ratio(pil_img, canvas_w, canvas_h)
                self.pil_images[key] = pil_img_resized
                self.tk_images[key] = ImageTk.PhotoImage(pil_img_resized)

                # 清空Canvas并显示调整后的图片
                target_canvas.delete("all")
                target_canvas.create_image(
                    canvas_w / 2, canvas_h / 2,
                    image=self.tk_images[key], anchor=tk.CENTER
                )

    # -----------------------------------------------------------------
    def detect_fruit(self):
        """按照改良后的算法，对图像进行水果检测和计数，并在GUI中显示各处理阶段的图像"""
        if self.cv_image is None:
            messagebox.showwarning("警告", "请先加载图片～")
            return

        # 获取当前图片的相对路径（相对于 FruitImages 文件夹）
        scenario = self.scenario_var.get()
        filename = os.path.basename(self.image_path)
        relative_path = f"{scenario}/{filename}"

        # 检查地面真实数据中是否存在该图片
        if relative_path not in self.ground_truth:
            messagebox.showerror("错误", f"地面真实数据中缺少图片 {relative_path} 的记录！")
            return

        # 地面真实数量
        true_counts = self.ground_truth[relative_path]

        # ============== 1) 颜色空间转换与阈值分割 ==============
        img = self.cv_image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 显示HSV转换后的图像
        hsv_rgb = cv2.cvtColor(hsv_blurred, cv2.COLOR_HSV2BGR)
        hsv_pil = Image.fromarray(cv2.cvtColor(hsv_rgb, cv2.COLOR_BGR2RGB))
        hsv_resized = self.resize_image_keep_ratio(hsv_pil, self.canvas_hsv.winfo_width(),
                                                   self.canvas_hsv.winfo_height())
        self.pil_images["hsv"] = hsv_resized
        self.tk_images["hsv"] = ImageTk.PhotoImage(hsv_resized)
        self.canvas_hsv.delete("all")
        self.canvas_hsv.create_image(
            self.canvas_hsv.winfo_width() / 2, self.canvas_hsv.winfo_height() / 2,
            image=self.tk_images["hsv"], anchor=tk.CENTER
        )

        # ============== 2) 定义水果的HSV颜色范围并创建掩膜 ==============
        fruit_colors = {
            "Apple": {
                "lower1": np.array([0, 100, 50]),  # 红色下限1
                "upper1": np.array([10, 255, 255]),  # 红色上限1
                "lower2": np.array([160, 100, 50]),  # 红色下限2
                "upper2": np.array([180, 255, 255]),  # 红色上限2
            },
            "Banana": {
                "lower": np.array([20, 100, 100]),  # 黄色下限
                "upper": np.array([30, 255, 255]),  # 黄色上限
            },
            "Orange": {
                "lower": np.array([10, 100, 100]),  # 橙色下限
                "upper": np.array([40, 255, 255]),  # 橙色上限（调整上限至20，避免与香蕉H=20重叠）
            },
        }

        # 创建各水果的掩膜并显示整体掩膜
        masks = {}
        for fruit, color in fruit_colors.items():
            if fruit == "Apple":
                mask1 = cv2.inRange(hsv_blurred, color["lower1"], color["upper1"])
                mask2 = cv2.inRange(hsv_blurred, color["lower2"], color["upper2"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_blurred, color["lower"], color["upper"])
            # 形态学操作去噪
            kernel = np.ones((3, 3), np.uint8)  # 调整核大小
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # 调整迭代次数
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            masks[fruit] = mask

        # 合并所有掩膜
        mask_all = cv2.bitwise_or(masks["Apple"], masks["Banana"])
        mask_all = cv2.bitwise_or(mask_all, masks["Orange"])

        # 显示合并后的颜色掩膜图像
        mask_all_rgb = cv2.cvtColor(mask_all, cv2.COLOR_GRAY2BGR)
        mask_pil = Image.fromarray(cv2.cvtColor(mask_all_rgb, cv2.COLOR_BGR2RGB))
        mask_resized = self.resize_image_keep_ratio(mask_pil, self.canvas_masks.winfo_width(),
                                                    self.canvas_masks.winfo_height())
        self.pil_images["masks"] = mask_resized
        self.tk_images["masks"] = ImageTk.PhotoImage(mask_resized)
        self.canvas_masks.delete("all")
        self.canvas_masks.create_image(
            self.canvas_masks.winfo_width() / 2, self.canvas_masks.winfo_height() / 2,
            image=self.tk_images["masks"], anchor=tk.CENTER
        )

        # ============== 3) 形态学处理：腐蚀 + 膨胀 ==============
        kernel = np.ones((3, 3), np.uint8)  # 调整核大小
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel, iterations=1)  # 调整迭代次数
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, kernel, iterations=1)

        # 显示形态学处理后的掩膜
        mask_morph_rgb = cv2.cvtColor(mask_all, cv2.COLOR_GRAY2BGR)
        mask_morph_pil = Image.fromarray(cv2.cvtColor(mask_morph_rgb, cv2.COLOR_BGR2RGB))
        mask_morph_resized = self.resize_image_keep_ratio(mask_morph_pil, self.canvas_morph.winfo_width(),
                                                          self.canvas_morph.winfo_height())
        self.pil_images["morph"] = mask_morph_resized
        self.tk_images["morph"] = ImageTk.PhotoImage(mask_morph_resized)
        self.canvas_morph.delete("all")
        self.canvas_morph.create_image(
            self.canvas_morph.winfo_width() / 2, self.canvas_morph.winfo_height() / 2,
            image=self.tk_images["morph"], anchor=tk.CENTER
        )

        # ============== 4) 粘连目标的分割（分水岭算法） ==============
        # 距离变换
        dist_transform = cv2.distanceTransform(mask_all, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(mask_all, sure_fg)

        # 标记连通组件
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # 所有标记增加1，背景为1
        markers[unknown == 255] = 0  # 未知区域标记为0

        # 计算梯度图以提升分水岭效果
        # 使用Sobel算子计算梯度，而不是Canny边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.sqrt(sobelx ** 2 + sobely ** 2)
        gradient = cv2.convertScaleAbs(gradient)

        # 显示梯度图像
        gradient_rgb = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        gradient_pil = Image.fromarray(cv2.cvtColor(gradient_rgb, cv2.COLOR_BGR2RGB))
        gradient_resized = self.resize_image_keep_ratio(gradient_pil, self.canvas_gradient.winfo_width(),
                                                        self.canvas_gradient.winfo_height())
        self.pil_images["gradient"] = gradient_resized
        self.tk_images["gradient"] = ImageTk.PhotoImage(gradient_resized)
        self.canvas_gradient.delete("all")
        self.canvas_gradient.create_image(
            self.canvas_gradient.winfo_width() / 2, self.canvas_gradient.winfo_height() / 2,
            image=self.tk_images["gradient"], anchor=tk.CENTER
        )

        # 应用分水岭算法，使用梯度图像
        markers = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)

        # 获取所有标签
        unique_labels = np.unique(markers)
        fruit_labels = [lab for lab in unique_labels if lab not in [0, 1, -1]]  # 排除背景和边界

        # 显示分水岭标记图
        markers_display = np.zeros_like(markers, dtype=np.uint8)
        markers_display[markers > 1] = 255  # 显示前景
        markers_rgb = cv2.cvtColor(markers_display, cv2.COLOR_GRAY2BGR)
        markers_pil = Image.fromarray(cv2.cvtColor(markers_rgb, cv2.COLOR_BGR2RGB))
        markers_resized = self.resize_image_keep_ratio(markers_pil, self.canvas_markers.winfo_width(),
                                                       self.canvas_markers.winfo_height())
        self.pil_images["markers"] = markers_resized
        self.tk_images["markers"] = ImageTk.PhotoImage(markers_resized)
        self.canvas_markers.delete("all")
        self.canvas_markers.create_image(
            self.canvas_markers.winfo_width() / 2, self.canvas_markers.winfo_height() / 2,
            image=self.tk_images["markers"], anchor=tk.CENTER
        )

        # ============== 5) 特征提取与分类 ==============
        fruit_dict = {"Apple": 0, "Banana": 0, "Orange": 0}
        output_image = img.copy()

        for lab in fruit_labels:
            mask_lab = np.where(markers == lab, 255, 0).astype(np.uint8)
            cnts_lab, _ = cv2.findContours(mask_lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts_lab) == 0:
                continue
            c_max = max(cnts_lab, key=cv2.contourArea)

            # 计算面积、周长和圆形度
            area = cv2.contourArea(c_max)
            perimeter = cv2.arcLength(c_max, True)
            circle_metric = 4.0 * np.pi * area / (perimeter * perimeter + 1e-5)

            x, y, w, h = cv2.boundingRect(c_max)

            # 计算ROI的HSV平均值
            roi_hsv = hsv[y: y + h, x: x + w]
            roi_mask = mask_lab[y: y + h, x: x + w]
            if roi_hsv.ndim == 3:
                nonzero_pixels = roi_hsv[roi_mask > 0]
            else:
                nonzero_pixels = None

            if nonzero_pixels is not None and len(nonzero_pixels) > 0:
                mean_h, mean_s, mean_v = np.mean(nonzero_pixels, axis=0)
            else:
                continue

            # 分类逻辑
            fruit_type = "Unknown"
            for fruit, color in fruit_colors.items():
                if fruit == "Apple":
                    if ((color["lower1"][0] <= mean_h <= color["upper1"][0]) or
                        (color["lower2"][0] <= mean_h <= color["upper2"][0])) and mean_s > 100:
                        fruit_type = "Apple"
                        break
                else:
                    if color["lower"][0] <= mean_h <= color["upper"][0] and mean_s > 100:
                        fruit_type = fruit
                        break

            if fruit_type == "Unknown":
                continue

            # 面积和圆形度过滤
            aspect_ratio = max(w / h, h / w)
            if fruit_type == "Banana":
                if area < 800 or aspect_ratio < 1.5:
                    continue
            elif fruit_type == "Apple":
                if area < 500 or aspect_ratio < 1.0:  # 调整苹果的面积和纵横比
                    continue
            elif fruit_type == "Orange":
                if area < 400 or aspect_ratio < 0.8:  # 调整橘子的面积和纵横比
                    continue

            fruit_dict[fruit_type] += 1

            # 在输出图像上绘制矩形和文本
            if fruit_type == "Apple":
                color_draw = (0, 0, 255)  # 红色
            elif fruit_type == "Banana":
                color_draw = (0, 255, 255)  # 黄色
            elif fruit_type == "Orange":
                color_draw = (0, 165, 255)  # 橙色 (BGR格式)
            else:
                color_draw = (0, 255, 0)  # 绿色（未知用）

            cv2.rectangle(output_image, (x, y), (x + w, y + h), color_draw, 2)
            cv2.putText(
                output_image, fruit_type, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_draw, 2
            )

        # ============== 6) 识别结果显示 ==============
        result_str = "识别结果：\n"
        for k, v in fruit_dict.items():
            if v > 0:
                result_str += f"{k} 数量: {v}\n"
        self.label_result.config(text=result_str)

        # 将处理后的图像显示在识别结果Canvas上
        show_img = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        show_pil = Image.fromarray(show_img)
        show_resized = self.resize_image_keep_ratio(show_pil, self.canvas_output.winfo_width(),
                                                    self.canvas_output.winfo_height())
        self.pil_images["output"] = show_resized
        self.tk_images["output"] = ImageTk.PhotoImage(show_resized)
        self.canvas_output.delete("all")
        self.canvas_output.create_image(
            self.canvas_output.winfo_width() / 2, self.canvas_output.winfo_height() / 2,
            image=self.tk_images["output"], anchor=tk.CENTER
        )

        # ============== 7) 性能指标计算 ==============
        # 获取检测结果
        detected_counts = fruit_dict

        # 初始化指标字典
        metrics = {}
        for fruit, true in true_counts.items():
            if true == 0:
                continue  # 仅处理数量不为零的水果
            detected = detected_counts.get(fruit, 0)
            TP = min(detected, true)  # True Positives
            FP = max(detected - true, 0)  # False Positives
            FN = max(true - detected, 0)  # False Negatives

            # 计算精确率
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

            # 计算召回率
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            # 计算F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[fruit] = {
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            }

        # 计算评估的水果类别数量
        evaluated_fruits_count = len(metrics)

        # 计算总体精确率、召回率和F1分数
        overall_precision = np.mean([m['Precision'] for m in metrics.values()]) if metrics else 0.0
        overall_recall = np.mean([m['Recall'] for m in metrics.values()]) if metrics else 0.0
        overall_f1 = np.mean([m['F1_Score'] for m in metrics.values()]) if metrics else 0.0

        # 显示性能指标
        if metrics:
            metrics_str = "性能评价：\n"
            for fruit, m in metrics.items():
                metrics_str += (f"{fruit}:\n"
                                f"  TP: {m['TP']}\n"
                                f"  FP: {m['FP']}\n"
                                f"  FN: {m['FN']}\n"
                                f"  精确率: {m['Precision'] * 100:.2f}%\n"
                                f"  召回率: {m['Recall'] * 100:.2f}%\n"
                                f"  F1 分数: {m['F1_Score'] * 100:.2f}%\n")

            # 仅当评估的水果类别数量大于1时，显示总体性能指标
            if evaluated_fruits_count > 1:
                metrics_str += (f"\n总体精确率: {overall_precision * 100:.2f}%\n"
                                f"总体召回率: {overall_recall * 100:.2f}%\n"
                                f"总体F1分数: {overall_f1 * 100:.2f}%")

            messagebox.showinfo("性能评价", metrics_str)
        else:
            messagebox.showinfo("性能评价", "没有需要显示的性能指标。")

        # ============== 8) 调试信息显示（可选） ==============
        for fruit, mask in masks.items():
            white_pixels = cv2.countNonZero(mask)
            print(f"{fruit} Mask White Pixels: {white_pixels}")

        for fruit, count in fruit_dict.items():
            print(f"Detected {fruit}: {count}")


def main():
    root = tk.Tk()
    app = FruitDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
