import cv2
import numpy as np

class SampleImgTransformer:
    def __init__(self, image, bg_color=(255, 255, 255), bg_thresh=200):
        self.original_image = image
        self.bg_color = bg_color
        self.bg_thresh = bg_thresh
        self.mask_image = self._create_mask(image)
        self.modified_image = self._add_padding(image)
    
    def _add_padding(self, image, padding=10):
        return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=self.bg_color)
    
    def _create_mask(self, image):
        mask = cv2.inRange(image, np.array(self.bg_color) - self.bg_thresh, np.array(self.bg_color) + self.bg_thresh)
        return mask
    
    def add_noise(self, noise_type='gaussian', mean=0, var=10, salt_pepper_ratio=0.01):
        if noise_type == 'gaussian':
            noise = np.random.normal(mean, var ** 0.5, self.modified_image.shape).astype('uint8')
            self.modified_image = cv2.add(self.modified_image, noise)
        elif noise_type == 'salt_pepper':
            salt_pepper = np.random.choice([0, 255], self.modified_image.shape[:2], p=[1 - salt_pepper_ratio, salt_pepper_ratio])
            self.modified_image[salt_pepper == 255] = 255
            self.modified_image[salt_pepper == 0] = 0

    def apply_rotation(self, angle):
        center = (self.modified_image.shape[1] // 2, self.modified_image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        self.modified_image = cv2.warpAffine(self.modified_image, matrix, (self.modified_image.shape[1], self.modified_image.shape[0]))

    def adjust_brightness(self, scale=1.0):
        hsv_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 2] = np.clip(hsv_image[..., 2] * scale, 0, 255)
        self.modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def perspective_transform(self, max_angle=10):
        h, w = self.modified_image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx, dy = np.random.uniform(-max_angle, max_angle, size=4).reshape(2, 2)
        pts2 = pts1 + np.float32([[dx[0], dy[0]], [dx[1], dy[0]], [dx[0], dy[1]], [dx[1], dy[1]]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.modified_image = cv2.warpPerspective(self.modified_image, matrix, (w, h))

    def apply_transformations(self, noise_type=None, rotation_angle=0, brightness_scale=1.0, perspective_angle=0):
        if noise_type:
            self.add_noise(noise_type=noise_type)
        if rotation_angle:
            self.apply_rotation(rotation_angle)
        if brightness_scale != 1.0:
            self.adjust_brightness(brightness_scale)
        if perspective_angle:
            self.perspective_transform(perspective_angle)

    def reset_image(self):
        self.modified_image = self._add_padding(self.original_image)

