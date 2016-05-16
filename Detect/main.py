#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os


def validate_contour(contour, img):
    [x, y, w, h] = cv2.boundingRect(contour)
    iw, ih = img.shape[1], img.shape[0]
    if w > iw/5 and 35 < h < (ih*0.8) and abs(x+w/2-iw/2) < 20:
        return True
    else:
        return False


def captch_ex(file_name):
    show = True
    img = cv2.imread(file_name)
    img_copy = img.copy()
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, new_img = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(new_img, kernel, iterations=9)
    dilated_copy = dilated.copy()
    ret, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get contours
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if validate_contour(contour, img):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if show:
        windows = [['Оригинал', img_copy], ['Отфильтрованный', new_img], ['С обводкой', dilated_copy], ['Результат', img]]
        for name, image in windows:
            cv2.imshow(name, image)
            cv2.moveWindow(name, 100, 100)
            cv2.waitKey()
            cv2.destroyWindow(name)
folder = 'Примеры'
for pic in reversed(os.listdir(folder)):
    captch_ex(os.path.join(folder, pic))

