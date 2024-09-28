import cv2


class Text:
    def __init__(self, font, font_scale,  thickness, color=(255, 255, 255)):
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.color = color

    def put_in_center(self, text, img):
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.thickness)
        img_height, img_width, _ = img.shape
        x = (img_width - text_width) // 2
        y = (img_height + text_height) // 2
        cv2.putText(img, text, (x, y), self.font,
                    self.font_scale, self.color, self.thickness)
