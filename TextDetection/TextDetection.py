import cv2
import matplotlib.pyplot as plt


def first():
    def captch_ex(file_name):
        img = cv2.imread(file_name)

        img_final = cv2.imread(file_name)
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
        ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
        '''
                line  8 to 12  : Remove noisy portion 
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                             3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
        dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

        # for cv2.x.x
        plt.imshow(dilated)
        plt.show()
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

        # for cv3.x.x comment above line and uncomment line below

        #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Don't plot small false positives that aren't text
            if w < 35 and h < 35:
                continue

            # draw rectangle around contour on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            '''
            #you can crop image and send to OCR  , false detected will return no text :)
            cropped = img_final[y :y +  h , x : x + w]
    
            s = file_name + '/crop_' + str(index) + '.jpg' 
            cv2.imwrite(s , cropped)
            index = index + 1
    
            '''
        # write original image with added contours to disk
        plt.imshow(img)
        plt.show()
        # cv2.imshow('captcha_result', img)
        # cv2.waitKey()


    file_name = 'Data/ClassificationCity/images/5_original.jpg'
    captch_ex(file_name)


def second():
    import cv2
    import numpy as np
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Load image, convert to HSV format, define lower/upper ranges, and perform
    # color segmentation to create a binary mask
    image = cv2.imread('Data/ClassificationCity/images/20_original.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 218])
    upper = np.array([157, 54, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Create horizontal kernel and dilate to connect text characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilate = cv2.dilate(mask, kernel, iterations=5)

    # Find contours and filter using aspect ratio
    # Remove non-text contours by filling in the contour
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if ar < 5:
            cv2.drawContours(dilate, [c], -1, (0, 0, 0), -1)

    # Bitwise dilated image with mask, invert, then OCR
    result = 255 - cv2.bitwise_and(dilate, mask)
    data = pytesseract.image_to_string(result, lang='eng', config='--psm 6')
    print(data)

    plt.imshow(mask)
    plt.show()
    plt.imshow(dilate)
    plt.show()
    plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    second()