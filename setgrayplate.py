import cv2
import numpy as np
def detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = 5
    blurred = cv2.GaussianBlur(gray, (k,k), 2, 7, cv2.BORDER_DEFAULT)
    #cv2.imshow('blurred', blurred)
    '''
    開運算
    '''
    kernel = np.ones((23, 27), np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('opened',opened)
    opened = cv2.addWeighted(blurred, 2, opened, -1, 0)
    #cv2.imshow('opened', opened)
    '''
    分割 threshold
    '''
    ret, thresh = cv2.threshold(opened, 50, 300, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = cv2.Canny(thresh, 59, 66)
    #cv2.imshow('canny',edge)
    '''
    使用開閉運算讓照片連成一個整體
    '''
    k = 7
    kernel = np.ones((k,k), np.uint8)
    edge1 = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    edge2 = cv2.morphologyEx(edge1, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 2300:
            temp_contours.append(contour)
    car_plates = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        '''
        車牌在正常情況下，寬高比在 2 至 5.5 之間
        '''
        if aspect_ratio > 3 and aspect_ratio < 7:
            car_plates.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
            rect_vertices = np.int0(rect_vertices)
    if len(car_plates) == 1:
        for car_plate in car_plates:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
            cv2.rectangle(image, (row_min, col_min), (row_max, col_max), (0, 255, 0), 7)
            card = image[col_min:col_max, row_min:row_max, :]
            cv2.imshow("img", image)
        #cv2.imshow("card_img.jpg", card)
if __name__ == '__main__':
    image = cv2.imread('08.jpeg')
    image = cv2.resize(image,(1000,500))
    detect(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
