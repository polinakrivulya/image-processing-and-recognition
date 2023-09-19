import numpy as np
import cv2
import skimage.measure as measure
import skimage.morphology as morphology
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# образцы цветов
color_ex = np.array([
    [18, 20, 160],  # red
    [70, 40, 50],  # blue
    [12, 160, 190],  # yellow
    [30, 40, 60],  # black
])


def get_color(color):
    return np.argmin(np.array([
            euclidean(color, color_ex[0]),
            euclidean(color, color_ex[1]),
            euclidean(color, color_ex[2]),
            euclidean(color, color_ex[3]),
    ]))


def elems(img, show=False):
    gray_pic = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # бинаризация
    _, threshold = cv2.threshold(gray_pic, 100, 1, type=cv2.THRESH_BINARY)
    threshold = np.logical_not(threshold)

    # компоненты связности
    labels, num = measure.label(threshold, return_num=True)
    # выпуклая оболочка
    convex_img = np.zeros(threshold.shape)
    for i in range(1, num + 1):
        convex_img += (morphology.convex_hull_image(labels == i)).astype(np.int64)
    # эрозия
    convex_img = morphology.erosion(convex_img, morphology.star(5))
    convex_img = (convex_img * 255).astype(np.uint8)
    if show:
        plt.imshow(convex_img, cmap='gray')
    cnts = cv2.findContours(convex_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def build_contours(img, show=False, number_of_elems=False):
    # проверено, работает со всем кроме path четко
    cnts = elems(img)[0]
    color_pic = img.copy()
    figures = []  # массивы вершин шестиугольников
    p = []
    for c in cnts:
        p.append(cv2.arcLength(c, True))  # периметр
    min_p = np.median(p) * 0.85
    max_p = np.median(p) * 1.15

    for c in cnts:
        p = cv2.arcLength(c, True)
        if (p > min_p) and (p < max_p):
            angles = cv2.approxPolyDP(c, 0.025 * p, True)  # угловые точки
            flag = True
            if len(angles) == 6:
                coordinates_list = []  # приведем координаты вершин к более удобному формату
                for i in range(0, len(angles)):
                    coordinates_list.append([angles[i][0, 0], angles[i][0, 1]])
                coordinates = np.array(coordinates_list)
                for i in range(0, len(angles)):
                    side_start = coordinates[(i + 1) % len(angles)]
                    side_end = coordinates[i % len(angles)]
                    if np.abs(euclidean(side_start, side_end) - p / len(angles)) > 1 / 12 * p:
                        flag = False
                if flag:
                    # нанесем контуры и углы найденной фишки
                    cv2.drawContours(color_pic, [angles], -1, (0, 255, 0), 2)
                    for n, i in enumerate(angles):
                        cv2.circle(color_pic, (i[0, 0], i[0, 1]), 5, (0, 255, 0), 2)
                    figures.append(coordinates)
    if number_of_elems:
        print(len(figures))
    return figures


def number_of_elem(obj, img_color_1):
    def get_color_notblack(color):
        return np.argmin(np.array([
            euclidean(color, color_ex[0]),  # red
            euclidean(color, color_ex[1]),  # blue
            euclidean(color, color_ex[2]),  # yellow
        ]))

    def new_color(old_color_list, new_color_num, old_color_num=-1):
        if old_color_num == 2:
            color_ex_local = np.array([
                [18, 20, 160],  # red
                [127, 137, 155],  # blue
                [12, 170, 210],  # yellow
                [30, 40, 60],  # black
            ])
        else:
            color_ex_local = color_ex
        len_blue = 100000
        num_len_blue = 0
        for i in old_color_list:
            # ищем точку примерно в центре ребра
            edge_center = (a * obj[(i + 1) % 6] + b * obj[i]).astype(np.uint64)
            # немного сдвигаемся к центру фишки
            curr_dot = (c * edge_center + d * center).astype(np.uint64)
            # фиксируем цвет найденной точки
            curr_col = img_color[curr_dot[1]][curr_dot[0]]
            col_ind = euclidean(curr_col, color_ex_local[new_color_num])
            if col_ind < len_blue:
                len_blue = col_ind
                num_len_blue = i
        return num_len_blue

    def perimetr_color(img, number):
        new_pic = img.copy()
        for k in range(img.shape[0]):  # все кроме number делаем черным
            for m in range(img.shape[1]):
                if new_pic[k][m][0] == new_pic[k][m][1] == new_pic[k][m][2] == 255:
                    new_pic[k][m] = [255, 255, 255]
                elif get_color(new_pic[k][m]) != number:
                    new_pic[k][m] = [255, 255, 255]
                else:
                    new_pic[k][m] = [0, 0, 0]
        cnts = elems(new_pic)[0]
        p_number = 0  # макс. площадь сегмента
        for k in cnts:
            p = cv2.arcLength(k, True)
            if p > p_number:
                p_number = p
        return p_number

    img_color = img_color_1.copy()
    center = np.mean(obj, axis=0)
    colors = []
    coeff_arr = np.array([[0.5, 0.5, 0.9, 0.1],
                          [0.5, 0.5, 1.0, 0.0],
                          [0.4, 0.6, 1.0, 0.0],
                          [0.6, 0.4, 1.0, 0.0],
                          [0.4, 0.6, 0.9, 0.1],
                          [0.6, 0.4, 0.9, 0.1]
                          ])
    for j in range(6):
        a = coeff_arr[j][0]
        b = coeff_arr[j][1]
        c = coeff_arr[j][2]
        d = coeff_arr[j][3]
        for i in range(6):  # по ребрам фигуры
            # ищем точку примерно в центре ребра
            edge_center = (a * obj[(i + 1) % 6] + b * obj[i]).astype(np.uint64)
            # к центру фишки
            curr_dot = (c * edge_center + d * center).astype(np.uint64)
            # фиксируем цвет найденной точки
            curr_col = img_color[curr_dot[1]][curr_dot[0]]
            col_ind = get_color_notblack(curr_col)
            colors.append(col_ind)

        red = []
        blue = []
        yellow = []
        for i in range(6):
            if colors[i] == 0:
                red.append(i)
            elif colors[i] == 1:
                blue.append(i)
            else:
                yellow.append(i)

        while not (len(red) == len(blue) == len(yellow)):
            if (len(red) > 2) and (len(blue) < 2):
                colors[new_color(red, 1)] = 1
            elif (len(red) > 2) and (len(yellow) < 2):
                colors[new_color(red, 2)] = 2
            elif (len(yellow) > 2) and (len(blue) < 2):
                colors[new_color(yellow, 1, 2)] = 1
            elif (len(yellow) > 2) and (len(red) < 2):
                colors[new_color(yellow, 0, 2)] = 0
            elif (len(blue) > 2) and (len(yellow) < 2):
                colors[new_color(blue, 2)] = 2
            elif (len(blue) > 2) and (len(red) < 2):
                colors[new_color(blue, 0)] = 0
            red = []
            blue = []
            yellow = []
            for i in range(6):
                if colors[i] == 0:
                    red.append(i)
                elif colors[i] == 1:
                    blue.append(i)
                else:
                    yellow.append(i)

        # самое веселое - определяем длины элементов и цвета...
        d = dict.fromkeys(['red', 'blue', 'yellow'])
        for i in range(len(colors)):
            if colors[i] == colors[i - 1]:
                if colors[i] == 0:
                    d['red'] = 'short'
                elif colors[i] == 1:
                    d['blue'] = 'short'
                else:
                    d['yellow'] = 'short'

            elif colors[i] == colors[i - 2]:
                if colors[i] == 0:
                    d['red'] = 'long'
                elif colors[i] == 1:
                    d['blue'] = 'long'
                else:
                    d['yellow'] = 'long'

            elif colors[i] == colors[i - 3]:
                if colors[i] == 0:
                    d['red'] = 'straight'
                elif colors[i] == 1:
                    d['blue'] = 'straight'
                else:
                    d['yellow'] = 'straight'

        num = -1
        if not ((d['red'] is None) or (d['yellow'] is None) or (d['blue'] is None)):
            # 1,10: длинный красный, длинный синий, короткий желтый
            if (d['red'] == 'long') and (d['blue'] == 'long') and (d['yellow'] == 'short'):
                # 1, 10
                x_1 = max(obj[:, 0])
                y_1 = max(obj[:, 1])
                x_0 = min(obj[:, 0])
                y_0 = min(obj[:, 1])
                new_pic = img_color[y_0:y_1, x_0:x_1, :]
                p_red = perimetr_color(new_pic, 0)
                p_blue = perimetr_color(new_pic, 1)
                if p_red > p_blue:
                    num = 1
                else:
                    num = 10

            # 7,8: длинный красный, длинный желтый, короткий синий
            elif (d['red'] == 'long') and (d['yellow'] == 'long') and (d['blue'] == 'short'):
                # 7, 8
                x_1 = max(obj[:, 0])
                y_1 = max(obj[:, 1])
                x_0 = min(obj[:, 0])
                y_0 = min(obj[:, 1])
                new_pic = img_color[y_0:y_1, x_0:x_1, :]
                p_red = perimetr_color(new_pic, 0)
                p_yellow = perimetr_color(new_pic, 2)
                if p_red > p_yellow:
                    num = 7
                else:
                    num = 8
            elif (d['red'] == 'short') and (d['blue'] == 'straight') and (d['yellow'] == 'short'):
                num = 2
            elif (d['red'] == 'short') and (d['blue'] == 'short') and (d['yellow'] == 'short'):
                num = 3
            elif (d['red'] == 'long') and (d['blue'] == 'straight') and (d['yellow'] == 'long'):
                num = 4
            elif (d['red'] == 'straight') and (d['blue'] == 'short') and (d['yellow'] == 'short'):
                num = 5
            elif (d['red'] == 'long') and (d['blue'] == 'long') and (d['yellow'] == 'straight'):
                num = 6
            elif (d['red'] == 'straight') and (d['blue'] == 'long') and (d['yellow'] == 'long'):
                num = 9
        if num != -1:
            return num
    return num


def white_cont(img):
    gray_pic = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_pic, 100, 1, type=cv2.THRESH_BINARY)
    threshold = np.logical_not(threshold)
    labels, num = measure.label(threshold, return_num=True)
    convex_img = np.zeros(threshold.shape)
    for i in range(1, num + 1):
        convex_img += (morphology.convex_hull_image(labels == i)).astype(np.int64)
    # эрозия
    convex_img = morphology.erosion(convex_img, morphology.star(5))
    convex_img = (convex_img * 255).astype(np.uint8)
    return cv2.bitwise_not(convex_img)


def group_task(img_color):
    img_color_copy = img_color.copy()
    kernel_open = np.ones((2, 2), np.uint8)
    photo_1 = cv2.morphologyEx(img_color_copy, cv2.MORPH_OPEN, kernel_open)

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(1, 2))
    lab = cv2.cvtColor(photo_1, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    photo_2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

    kernel_close = np.ones((1, 1), np.uint8)
    photo_3 = cv2.morphologyEx(photo_2, cv2.MORPH_CLOSE, kernel_close)

    obj_list_1 = build_contours(photo_1, show=False)
    obj_list_2 = build_contours(photo_2, show=False)
    obj_list_3 = build_contours(photo_3, show=False)

    mask = white_cont(photo_1)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    photo_1 = cv2.bitwise_or(photo_1, mask)

    mask = white_cont(photo_2)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    photo_2 = cv2.bitwise_or(photo_2, mask)

    mask = white_cont(photo_3)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    photo_3 = cv2.bitwise_or(photo_3, mask)

    for i in range(len(obj_list_1)):
        shape_num_1 = number_of_elem(obj_list_1[i], photo_1)
        shape_num_2 = number_of_elem(obj_list_2[i], photo_2)
        shape_num_3 = number_of_elem(obj_list_3[i], photo_3)
        lst = [shape_num_1, shape_num_2, shape_num_3]
        shape_num = Counter(lst).most_common(1)[0][0]
        cv2.putText(img_color_copy, str(shape_num), org=(obj_list_1[i][0, 0] + 5, obj_list_1[i][0, 1] + 5),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=[0, 255, 0], thickness=3)
    return img_color_copy


def single_task(img_color):
    img_color_copy = img_color.copy()
    kernel_open = np.ones((2, 2), np.uint8)
    photo_1 = cv2.morphologyEx(img_color_copy, cv2.MORPH_OPEN, kernel_open)
    # контраст
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(1, 2))
    lab = cv2.cvtColor(photo_1, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    photo_2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

    kernel_close = np.ones((1, 1), np.uint8)
    photo_3 = cv2.morphologyEx(photo_2, cv2.MORPH_CLOSE, kernel_close)

    obj_list_1 = build_contours(photo_1, show=False)
    obj_list_2 = build_contours(photo_2, show=False)
    obj_list_3 = build_contours(photo_3, show=False)

    mask = white_cont(photo_1)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    photo_1 = cv2.bitwise_or(photo_1, mask)

    mask = white_cont(photo_2)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    photo_2 = cv2.bitwise_or(photo_2, mask)

    mask = white_cont(photo_3)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    photo_3 = cv2.bitwise_or(photo_3, mask)

    shape_num_1 = number_of_elem(obj_list_1[0], photo_1)
    shape_num_2 = number_of_elem(obj_list_2[0], photo_2)
    shape_num_3 = number_of_elem(obj_list_3[0], photo_3)
    lst = [shape_num_1, shape_num_2, shape_num_3]

    shape_num = Counter(lst).most_common(1)[0][0]
    return shape_num


print("Введите номер задачи: 1 для изображения Single, 2 для изображения Group")
num = int(input())
if num == 2:
    print("Введите имя файла (Group_*.bmp). Он должен находиться в той же папке, что и программа.")
    name = input()
    photo = cv2.imread(name, cv2.IMREAD_COLOR)
    img = group_task(photo)
    cv2.imshow(name, img)
    cv2.waitKey(10)
    cv2.destroyAllWindows()
elif num == 1:
    print("Введите имя файла (Single_*.bmp). Он должен находиться в той же папке, что и программа.")
    name = input()
    photo = cv2.imread(name, cv2.IMREAD_COLOR)
    print("Номер фишки ", single_task(photo))
else:
    print("Введен неправильный номер задачи")

print("Введите что-нибудь если конец")
_ = input()
cv2.destroyAllWindows()
