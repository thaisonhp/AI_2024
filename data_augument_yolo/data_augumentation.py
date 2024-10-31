import cv2
import numpy as np

class SampleImgTransformer:
    def __init__(self, image: np.ndarray, bg_color: tuple = (255, 255, 255), bg_thresh: int = 200):
        """
        Initializes the SampleImgTransformer class.

        Args:
            image (np.ndarray): Input image to be transformed.
            bg_color (tuple): Background color used for masking and padding, default is white (255, 255, 255).
            bg_thresh (int): Threshold for background color range in mask creation.
        """
        self.original_image = image
        self.bg_color = bg_color
        self.bg_thresh = bg_thresh
        self.mask_image = self._create_mask(image)
        self.modified_image = self._add_padding(image)
    
    def _add_padding(self, image: np.ndarray, padding: int = 10) -> np.ndarray:
        """
        Adds padding around the image.

        Args:
            image (np.ndarray): Image to add padding to.
            padding (int): Amount of padding to add on each side.

        Returns:
            np.ndarray: Image with added padding.
        """
        return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=self.bg_color)
    
    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask for areas in the image matching a specific color range.

        Args:
            image (np.ndarray): Input image.
        
        Returns:
            np.ndarray: Binary mask highlighting areas in the specified color range.
        """
        mask = cv2.inRange(image, np.array(self.bg_color) - self.bg_thresh, np.array(self.bg_color) + self.bg_thresh)
        return mask
    
    def add_noise(self, noise_type: str = 'gaussian', mean: int = 0, var: int = 10, salt_pepper_ratio: float = 0.01) -> None:
        """
        Adds noise to the image.

        Args:
            noise_type (str): Type of noise to add ('gaussian' or 'salt_pepper').
            mean (int): Mean of Gaussian noise.
            var (int): Variance of Gaussian noise.
            salt_pepper_ratio (float): Ratio for salt and pepper noise.
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(mean, var ** 0.5, self.modified_image.shape).astype('uint8')
            self.modified_image = cv2.add(self.modified_image, noise)
        elif noise_type == 'salt_pepper':
            salt_pepper = np.random.choice([0, 255], self.modified_image.shape[:2], p=[1 - salt_pepper_ratio, salt_pepper_ratio])
            self.modified_image[salt_pepper == 255] = 255
            self.modified_image[salt_pepper == 0] = 0

    def apply_rotation(self, angle: float) -> None:
        """
        Rotates the image by a given angle.

        Args:
            angle (float): The angle by which to rotate the image.
        """
        center = (self.modified_image.shape[1] // 2, self.modified_image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        self.modified_image = cv2.warpAffine(self.modified_image, matrix, (self.modified_image.shape[1], self.modified_image.shape[0]))

    def adjust_brightness(self, scale: float = 1.0) -> None:
        """
        Adjusts the brightness of the image.

        Args:
            scale (float): Scaling factor for brightness. Values greater than 1 increase brightness.
        """
        hsv_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 2] = np.clip(hsv_image[..., 2] * scale, 0, 255)
        self.modified_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def perspective_transform(self, max_angle: float = 10) -> None:
        """
        Applies a perspective transformation to the image.

        Args:
            max_angle (float): Maximum angle of transformation.
        """
        h, w = self.modified_image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx, dy = np.random.uniform(-max_angle, max_angle, size=4).reshape(2, 2)
        pts2 = pts1 + np.float32([[dx[0], dy[0]], [dx[1], dy[0]], [dx[0], dy[1]], [dx[1], dy[1]]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.modified_image = cv2.warpPerspective(self.modified_image, matrix, (w, h))

    def apply_scaling(self, scale_factor: float) -> None:
        """
        Scales the image by a given factor.

        Args:
            scale_factor (float): The factor by which to scale the image.
        """
        h, w = self.modified_image.shape[:2]
        self.modified_image = cv2.resize(self.modified_image, (int(w * scale_factor), int(h * scale_factor)))

    def flip_image(self, direction: str = 'horizontal') -> None:
        """
        Flips the image horizontally or vertically.

        Args:
            direction (str): 'horizontal' or 'vertical'.
        """
        if direction == 'horizontal':
            self.modified_image = cv2.flip(self.modified_image, 1)
        elif direction == 'vertical':
            self.modified_image = cv2.flip(self.modified_image, 0)

    def crop_image(self, x: int, y: int, width: int, height: int) -> None:
        """
        Crops the image to a specified rectangle.

        Args:
            x (int): x-coordinate of the top-left corner.
            y (int): y-coordinate of the top-left corner.
            width (int): Width of the rectangle to crop.
            height (int): Height of the rectangle to crop.
        """
        self.modified_image = self.modified_image[y:y + height, x:x + width]
    
    def adjust_contrast(self, alpha: float) -> None:
        """
        Adjusts the contrast of the image.

        Args:
            alpha (float): Contrast control (1.0 = no change, <1.0 = lower contrast, >1.0 = higher contrast).
        """
        self.modified_image = cv2.convertScaleAbs(self.modified_image, alpha=alpha, beta=0)


    def show_image(self , action : str , image):
        '''
        Show the image 
        '''
        cv2.imshow(f"{action}" , image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_transformations(self, noise_type: str = None, rotation_angle: float = 0, brightness_scale: float = 1.0, perspective_angle: float = 0) -> None:
        """
        Applies a series of transformations to the image and displays the modified image after each transformation.

        Args:
            noise_type (str): Type of noise to add ('gaussian' or 'salt_pepper').
            rotation_angle (float): Angle for rotatdion.
            brightness_scale (float): Scaling factor for brightness.
            perspective_angle (float): Maximum angle for perspective transformation.
        """
        if noise_type:
            self.add_noise(noise_type=noise_type)
            self.show_image(action="add_noise", image=self.modified_image)
        
        if rotation_angle:
            self.apply_rotation(rotation_angle)
            self.show_image(action="rotate", image=self.modified_image)
        
        if brightness_scale != 1.0:
            self.adjust_brightness(brightness_scale)
            self.show_image(action="adjust_brightness", image=self.modified_image)
        
        if perspective_angle:
            self.perspective_transform(perspective_angle)
            self.show_image(action="perspdective_transform", image=self.modified_image)

    def reset_image(self) -> None:
        """
        Resets the image to the original with padding added.
        """
        self.modified_image = self._add_padding(self.original_image)
