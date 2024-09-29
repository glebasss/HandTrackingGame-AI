import cv2


class Text:
    def __init__(self, font, font_scale,  thickness, color=(255, 255, 255)):
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.color = color

    def put_in_center(self, text_list, img):
        h, w, _ = img.shape
        text_height = cv2.getTextSize(
            text_list[0], self.font, self.font_scale, self.thickness)[0][1]
        text_positions = []
        # Расчет позиции для каждого текста
        y_offset = h // 2 - (len(text_list) * text_height) // 2
        for idx, text in enumerate(text_list):
            text_size = cv2.getTextSize(
                text, self.font, self.font_scale, self.thickness)[0]
            x = (w - text_size[0]) // 2  # Центрирование по оси X
            # Расположение по оси Y с отступом
            y = y_offset + idx * (text_size[1] + 40)
            cv2.putText(img, text, (x, y), self.font,
                        self.font_scale, self.color, self.thickness)
            # Сохраняем координаты текста для взаимодействия
            text_positions.append((x, y - text_size[1], x + text_size[0], y))

        return img, text_positions


class Rectangle:
    def create_rectangle(self, coord, img):
        '''
        coord : x1,y1,x2,y2 [100,105,130,110]
        '''
        coord_copy = list(coord).copy()
        x1, y1, x2, y2 = coord_copy
        x1, y1, x2, y2 = x1-10, y1-10, x2+10, y2+10
        cv2.line(img, (x1, y1), (x2, y1), (0, 0, 0), 4)
        cv2.line(img, (x1, y1), (x1, y2), (0, 0, 0), 4)
        cv2.line(img, (x2, y2), (x1, y2), (0, 0, 0), 4)
        cv2.line(img, (x2, y2), (x2, y1), (0, 0, 0), 4)
        cv2.rectangle(img, (x1+2, y1+2), (x2-2, y2-2), (255, 255, 255), -1)
        return x1, y1, x2, y2
