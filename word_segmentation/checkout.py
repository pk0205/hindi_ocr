"""
For Testing purposes
    Take image from user, crop the background and transform perspective
    from the perspective detect the word and return the array of word's
    bounding boxes
"""
from word_segmentation import page
from word_segmentation import words

from PIL import Image
import cv2
import numpy as np
from pythonRLSA import rlsa
from pathlib import Path
import matplotlib.pyplot as plt


def show_img(image, win_name):
    cv2.imshow(win_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def otsu(image, is_normalized=True):
    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * \
        weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]

    return threshold


def segment(path):

    # User input page image
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    '''
    
    cv2.namedWindow('', cv2.WINDOW_NORMAL)
    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    '''

    # Crop image and get bounding boxes
    crop = page.detection(image)

    '''
    cv2.namedWindow('', cv2.WINDOW_NORMAL)
    cv2.imshow('', crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    boxes = words.detection(crop)

    lines = words.sort_words(boxes)

    # Saving the bounded words from the page image in sorted way
    i = 11000
    for line in lines:
        text = crop.copy()
        for (x1, y1, x2, y2) in line:
            # roi = text[y1:y2, x1:x2]
            save = Image.fromarray(text[y1:y2, x1:x2])

            thresh = otsu(save)

            def fn(x): return 255 if x > thresh else 0
            r = save.convert('L').point(fn, mode='1')
            # r.save('foo.png')

            #retval, binary = cv2.threshold(save, threshold, 255, cv2.THRESH_OTSU)
            # print(threshold)
            # print(i)
            r.save("test/Image" + str(i) + ".jpg")
            i += 1


def segment_line(path):

    # User input page image
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (0, 0), fx=1200 /
                       image.shape[0], fy=1200/image.shape[0])

    # Crop image and get bounding boxes
    crop = page.detection(image)

    (bw_img, image, join) = words.detection(crop)
    # show_img(bw_img,'bw_img')
    from pythonRLSA import rlsa

    image_rlsa = bw_img.copy()

    image_rlsa = ~image_rlsa

    rlsa.rlsa(image_rlsa, False, True, 20)

    #show_img(image_rlsa,'rlsa 1')

    rlsa.rlsa(image_rlsa, True, False, 200)
    #show_img(image_rlsa,'rlsa 2')

    bw_img = ~image_rlsa

    # show_img(bw_img,'bw_img')

    boxes = words._text_detect(bw_img, image, join)

    lines = words.sort_words(boxes)

    # Saving the bounded words from the page image in sorted way
    i = 1
    for line in lines:
        text = crop.copy()
        for (x1, y1, x2, y2) in line:
            # roi = text[y1:y2, x1:x2]
            save = Image.fromarray(text[y1:y2, x1:x2])

            thresh = otsu(save)

            def fn(x): return 255 if x > thresh else 0
            r = save.convert('L').point(fn, mode='1')

            r.save("ocr/static/lines/Image" + str(i) + ".jpg")
            i += 1


def segment_words(image, line_number):
    from pythonRLSA import rlsa
    image_rlsa = image.copy()

    rlsa.rlsa(image_rlsa, False, True, 30)

    #show_img(image_rlsa,'rlas 1')

    rlsa.rlsa(image_rlsa, True, False, 60)
    #show_img(image_rlsa,'rlas 2')

    image_rlsa_1 = image_rlsa
    image_rlsa_1 = ~image_rlsa_1
    contours = cv2.findContours(
        image_rlsa_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    num = 0
    imgs = []
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if(w < 20 or h < 20):
            continue
        imgs.append([x, y, w, h])
    imgs.sort()

    for img_box in imgs:
        (x, y, w, h) = img_box
        ROI = image[y:y+h, x:x+w]
        plt.imsave(
            f'ocr/static/words/Image{line_number}{num}.jpg', ROI, cmap='gray')
        # show_img(ROI)
        num = num+1


images = sorted(list(map(str, list(Path('./lines').glob("*.jpg")))))

line_number = 1

for img in images:
    image = cv2.imread(img, 0)
    segment_words(image, line_number)
    line_number = line_number + 1


def word_segmentor(filename):
    segment_line(filename)

    images = sorted(
        list(map(str, list(Path('ocr/static/lines/').glob("*.jpg")))))

    line_number = 1

    for img in images:
        image = cv2.imread(img, 0)
        segment_words(image, line_number)
        line_number = line_number + 1
