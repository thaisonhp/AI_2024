import cv2
from data_augumentation import SampleImgTransformer

# Đọc ảnh từ file
image = cv2.imread('/Users/luongthaison/Documents/Third_years_student/AI_AnhHung/YOLO_data_aug/data/vu.jpeg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Khởi tạo đối tượng với ảnh gốc
transformer = SampleImgTransformer(image)
cv2.imshow('Original Image 2', transformer.original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Xoay ảnh 90 độ
transformer.apply_rotation(angle=90)
cv2.imshow('Rotated Image', transformer.modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Thêm padding 50px
image_with_padding = transformer._add_padding(image=image, padding=50)
cv2.imshow('Image with Padding', image_with_padding)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Thêm noise Gaussian
transformer.add_noise(noise_type='gaussian')
cv2.imshow('Image with Gaussian Noise', transformer.modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Thay đổi độ sáng
transformer.adjust_brightness(scale=1.5)
cv2.imshow('Brightness Adjusted Image', transformer.modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Biến đổi phối cảnh
transformer.perspective_transform(max_angle=5)
cv2.imshow('Perspective Transformed Image', transformer.modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Lấy lại ảnh gốc
transformer.reset_image()
cv2.imshow('Reset Image', transformer.modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
