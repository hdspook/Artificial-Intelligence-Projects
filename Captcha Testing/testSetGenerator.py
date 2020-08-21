import cv2
import glob
import os
import imutils
import os.path


CAPTCHA_IMAGES = 'dataset'
TEST_IMAGES = 'trainingDataSet'

captchImageList = glob.glob(os.path.join(CAPTCHA_IMAGES, '*'))
counts = {}

for(i, singleCaptcha) in enumerate(captchImageList):
    print("Processing "+str((i+1))+" image of total: "+str(len(captchImageList)))
    fileName = os.path.basename(singleCaptcha)
    correctCaptcha = os.path.splitext(fileName)[0]
    
    image = cv2.imread(singleCaptcha)
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, bw_copy = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
     # bilateral filter
    blur = cv2.bilateralFilter(gray, 5, 75, 75)
     # cv2.imshow('blur', blur)
    
     # morphological gradient calculation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
     # cv2.imshow('gradient', grad)
    
     # binarization
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('otsu', bw)

    ###########################################
    ###########################################
    #########################
    #grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
  
    #(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
     
    #cv2.imshow('Black white image', blackAndWhiteImage)
    #cv2.imshow('Original image',originalImage)
    #cv2.imshow('Gray image', grayImage)
    ############################
    
    
    #grayImageSample = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("h",grayImageSample)
    
    #grayImageSample = cv2.copyMakeBorder(grayImageSample, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    #blackWhiteFormatted = cv2.threshold(grayImageSample, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("h=f",grayImageSample)
    #cv2.imshow('Gray image', blackWhiteFormatted)
    
    #fINDING CONTINOUS PIXELS OF DIGITS
    
    continuousPixels = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    continuousPixels = continuousPixels[1] if imutils.is_cv3() else continuousPixels[0]
    
    letters = []
    
    for singleDigits in continuousPixels:
        
        (x, y, w, h) = cv2.boundingRect(singleDigits)
        
        letters.append((x, y, w, h))
        letters = sorted(letters, key=lambda x:x[0])
        
    for letter_bounding_box, letter_text in zip(letters, correctCaptcha):
        x, y, w, h = letter_bounding_box
        letter_image = gray[y - 1:y + h + 1, x - 1:x + w + 1]
        save_path = os.path.join(TEST_IMAGES, letter_text)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1
        
        

        
        
    
    
    
    
    


