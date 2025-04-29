import os
import sys
import cv2
import time
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.base_dataset import BaseDataset, get_params, get_transform, normalize

from ui.ui import Ui_Form
from ui.mouse_event import GraphicsScene
from ui_util.config import Config

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0), QColor(51, 51, 255), QColor(204, 0, 204), QColor(0, 255, 255), QColor(51, 255, 255), QColor(102, 51, 0), QColor(255, 0, 0), QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0)]

class Ex(QWidget, Ui_Form):
    def __init__(self, model, opt):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()
        self.model = model
        self.opt = opt

        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.img = None

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.mode, self.size)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = QGraphicsScene()
        self.graphicsView_2.setScene(self.ref_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
 
        self.result_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.result_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(image)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(image)

    def open_mask(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        if fileName:
            mat_img = cv2.imread(fileName)
            mat_img = cv2.cvtColor(mat_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            self.mask = mat_img.copy()
            self.mask_m = mat_img.copy()

            label_color_map = {
                (0, 0, 0): 0, (204, 0, 0): 1, (76, 153, 0): 2, (204, 204, 0): 3, (51, 51, 255): 4,
                (204, 0, 204): 5, (0, 255, 255): 6, (51, 255, 255): 7, (102, 51, 0): 8, (255, 0, 0): 9,
                (102, 204, 0): 10, (255, 255, 0): 11, (0, 0, 153): 12, (0, 0, 204): 13, (255, 51, 153): 14,
                (0, 204, 204): 15, (0, 51, 0): 16, (255, 153, 51): 17, (0, 204, 0): 18
            }

            image = QImage(512, 512, QImage.Format_RGB888)
            for i in range(512):
                for j in range(512):
                    r, g, b = mat_img[j, i]
                    label_idx = label_color_map.get((r, g, b), 0)  # fallback to 0 if unknown
                    image.setPixel(i, j, color_list[label_idx].rgb())

            pixmap = QPixmap()
            pixmap.convertFromImage(image)
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items()) > 0:
                self.scene.reset_items()
            self.scene.addPixmap(self.image)

    def bg_mode(self):
        self.scene.mode = 0

    def skin_mode(self):
        self.scene.mode = 1

    def nose_mode(self):
        self.scene.mode = 2

    def eye_g_mode(self):
        self.scene.mode = 3

    def l_eye_mode(self):
        self.scene.mode = 4

    def r_eye_mode(self):
        self.scene.mode = 5

    def l_brow_mode(self):
        self.scene.mode = 6

    def r_brow_mode(self):
        self.scene.mode = 7

    def l_ear_mode(self):
        self.scene.mode = 8

    def r_ear_mode(self):
        self.scene.mode = 9

    def mouth_mode(self):
        self.scene.mode = 10

    def u_lip_mode(self):
        self.scene.mode = 11

    def l_lip_mode(self):
        self.scene.mode = 12

    def hair_mode(self):
        self.scene.mode = 13

    def hat_mode(self):
        self.scene.mode = 14

    def ear_r_mode(self):
        self.scene.mode = 15

    def neck_l_mode(self):
        self.scene.mode = 16

    def neck_mode(self):
        self.scene.mode = 17

    def cloth_mode(self):
        self.scene.mode = 18

    def increase(self):
        if self.scene.size < 15:
            self.scene.size += 1
    
    def decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1 

    def edit(self):
        import numpy as np
        import time
        import torch
        from PIL import Image
        from PyQt5.QtGui import QImage, QPixmap

        # Mapping of label indices to RGB values
        index_to_rgb = {
            0: (0, 0, 0), 1: (204, 0, 0), 2: (76, 153, 0), 3: (204, 204, 0), 4: (51, 51, 255),
            5: (204, 0, 204), 6: (0, 255, 255), 7: (51, 255, 255), 8: (102, 51, 0), 9: (255, 0, 0),
            10: (102, 204, 0), 11: (255, 255, 0), 12: (0, 0, 153), 13: (0, 0, 204), 14: (255, 51, 153),
            15: (0, 204, 204), 16: (0, 51, 0), 17: (255, 153, 51), 18: (0, 204, 0)
        }

        def rgb_to_label(mask_rgb):
            label_map = np.zeros((512, 512), dtype=np.uint8)
            color_to_index = {v: k for k, v in index_to_rgb.items()}
            for rgb, idx in color_to_index.items():
                dist = np.linalg.norm(mask_rgb - np.array(rgb), axis=-1)
                label_map[dist < 10] = idx
            return label_map

        def make_mask(mask, pts, sizes, color):
            if len(pts) > 0:
                for idx, pt in enumerate(pts):
                    cv2.line(mask, pt['prev'], pt['curr'], index_to_rgb[color], sizes[idx])
            return mask

        # Step 1: Apply mask edits with correct RGB colors
        for i in range(19):
            self.mask_m = make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)

        # Step 2: Preprocessing
        params = get_params(self.opt, (512, 512))
        transform_image = get_transform(self.opt, params)

        mask_label = rgb_to_label(self.mask.copy())     # reference mask
        mask_m_label = rgb_to_label(self.mask_m.copy()) # edited mask

        mask_tensor = torch.from_numpy(mask_label).unsqueeze(0).unsqueeze(0).long()
        mask_m_tensor = torch.from_numpy(mask_m_label).unsqueeze(0).unsqueeze(0).long()
        img_tensor = transform_image(self.img).unsqueeze(0).float()

        # Step 3: Inference
        start_t = time.time()
        generated = model.inference(mask_m_tensor, mask_tensor, img_tensor)
        end_t = time.time()
        print(f'Inference time: {end_t - start_t:.3f}s')

        # Step 4: Post-process and display
        generated = torch.clamp(generated, -1, 1)
        result = generated.permute(0, 2, 3, 1).cpu().numpy()[0]
        result = ((result + 1) * 127.5).astype(np.uint8)
        result = result.copy()

        self.output_img = result 

        qim = QImage(result.data, result.shape[1], result.shape[0], result.shape[1] * 3, QImage.Format_RGB888)
        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(QPixmap.fromImage(qim))


    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'],(color,color,color),sizes[idx])
        return mask

    def save_img(self):
        if self.output_img is not None and self.output_img.size != 0:
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File", QDir.currentPath())
            if fileName:
                # Convert RGB to BGR before saving
                bgr_image = cv2.cvtColor(self.output_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fileName + '.jpg', bgr_image)
        else:
            print("Error: output_img is empty. Cannot save.")

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.mask_m = self.mask.copy()
    
        self.scene.reset_items()
        self.scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)

if __name__ == '__main__':
    # Select device: GPU if available, else CPU
    if torch.cuda.is_available():
        device = 'cuda:0'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        device = 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"[INFO] Using device: {device}")

    # Set test-time options
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.isTrain = False

    # Path to pretrained generator
    opt.name = 'label2face_512p'
    opt.checkpoints_dir = '../MaskGAN_demo/checkpoints'

    # Load model
    model = create_model(opt)

    # Start GUI
    app = QApplication(sys.argv)
    ex = Ex(model, opt)
    sys.exit(app.exec_())