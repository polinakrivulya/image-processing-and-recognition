import numpy as np
import cv2
import skimage.morphology as morphology


def skelet_1(photo):
    _, threshold = cv2.threshold(photo, 80, 1, type=cv2.THRESH_BINARY_INV)
    photo_bin = morphology.binary_closing(threshold, morphology.star(10))
    skeleton = morphology.skeletonize(photo_bin).astype(np.uint8)
    skeleton[:60, :] = 0
    skeleton[-60:, :] = 0
    skeleton[:, :60] = 0
    skeleton[:, -60:] = 0
    skeleton = morphology.dilation(skeleton, morphology.disk(3))
    return skeleton


def skelet_2(photo_color):
    hsv = cv2.cvtColor(photo_color, cv2.COLOR_BGR2HSV)  # Преобразуем в HSV
    color_low = (33, 45, 35)
    color_high = (140, 220, 255)
    img_hsv = cv2.inRange(hsv, color_low, color_high)
    img_hsv = morphology.erosion(img_hsv, morphology.disk(1))
    img_hsv = morphology.dilation(img_hsv, morphology.disk(10))
    _, threshold = cv2.threshold(img_hsv, 80, 1, type=cv2.THRESH_BINARY)
    photo_bin = morphology.binary_closing(threshold, morphology.star(10))
    skeleton = morphology.skeletonize(photo_bin).astype(np.uint8)
    skeleton = morphology.dilation(skeleton, morphology.disk(5))
    skeleton[:60, :] = 0
    skeleton[-60:, :] = 0
    skeleton[:, :60] = 0
    skeleton[:, -60:] = 0
    return skeleton


def main(skeleton):
    p = []
    cnts, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_norm = []

    for i in range(len(cnts) - 1, -1, -1):  # убираем слишком маленькие контуры
        if cv2.arcLength(cnts[i], True) >= 200:
            cnts_norm.append(cnts[i])
    cnts = cnts_norm
    if len(cnts) == 6:
        return 3
    elif len(cnts) == 5:
        return 1
    for c in cnts:
        p.append(cv2.arcLength(c, True))  # периметр
    circles = []
    for c in cnts:  # поиск угловых точек
        p = cv2.arcLength(c, True)
        angles = cv2.approxPolyDP(c, 0.025 * p, True)  # угловые точки
        for n, i in enumerate(angles):  # не добавляем уже добавленные
            flag = False
            if len(circles):
                for k in circles:
                    if abs(k[0] - i[0, 0]) + abs(k[1] - i[0, 1]) < 100:
                        flag = True
                        break
            if not flag:
                circles.append([i[0, 0], i[0, 1]])
    stepeni = []
    for circle in circles:  # степени полученных углов
        newy_min = min(circle[0] - 48, skeleton.shape[0])
        newy_max = min(circle[0] + 48, skeleton.shape[0])
        newx_min = min(circle[1] - 48, skeleton.shape[1])
        newx_max = min(circle[1] + 48, skeleton.shape[1])
        new_pic = skeleton[newx_min: newx_max, newy_min: newy_max]
        new_pic = morphology.erosion(new_pic, morphology.star(2))
        new_pic = (new_pic * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(new_pic, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        coordinates_list = []
        for c in cnts:
            p = cv2.arcLength(c, True)
            angles = cv2.approxPolyDP(c, 0.025 * p, True)  # угловые точки
            for i in range(0, len(angles)):
                coordinates_list.append([angles[i][0, 0], angles[i][0, 1]])
        st = 0
        for i in coordinates_list:
            if abs(i[0] - new_pic.shape[1]) < 2 or abs(i[1] - new_pic.shape[0]) < 2 or \
                    abs(i[0]) < 2 or abs(i[1]) < 2:
                st += 1
        stepeni.append(st)

    if (np.array(stepeni) == 4).sum() >= 2:
        return 4
    else:
        return 2


print("Введите номер задачи: 1 для изображения на однотонном фоне, 2 для общего случая")
num = int(input())
print("Введите имя файла (*.jpg). "
      "Он должен находиться в той же папке, что и программа. Введите '0', чтобы прервать.")
name = input()
while name != '0':
    photo = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    photo_color = cv2.imread(name, cv2.IMREAD_COLOR)
    if num == 1:
        skeleton = skelet_1(photo)
    else:
        skeleton = skelet_2(photo_color)
    ans = main(skeleton)
    print('Граф типа ', ans)
    print("Введите имя файла. "
          "Он должен находиться в той же папке, что и программа. Введите '0', чтобы прервать.")
    name = input()
