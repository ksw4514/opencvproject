from bing_image_downloader import downloader
import os
import cv2
import numpy as np

def download(query_string, limit):
    downloader.download(query_string, limit,  output_dir='dataset', filter = 'jpg', force_replace=True, timeout=60, verbose=True)

class CompareImage(object):

    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 2
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path

    def compare_image(self):
        image_1 = cv2.imread(self.image_1_path, 0)
        image_2 = cv2.imread(self.image_2_path, 0)
        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            return commutative_image_diff
        return 10000 

    @ staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff

if __name__ == '__main__':

    topic = 'car'
    num = 10
    ##download(topic, num)

    for m in range (1,10):
        img_main_path = 'C:/opencvproject/main_image/Image_main_'+str(m)+'.jpg'
        for n in range (1,num+1):
            img_path = 'C:/opencvproject/dataset/'+topic+'/Image_'+str(n)+'.jpg'
            compare_image = CompareImage(img_main_path, img_path)       
            image_difference = compare_image.compare_image()
            print('main_'+str(m)+' | comparison_'+str(n))
            print (image_difference)
