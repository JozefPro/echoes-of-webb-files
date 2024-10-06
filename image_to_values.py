import cv2
import numpy as np

def calc(url):
    img = cv2.imread(url)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mean_brightness = hsv[:, :, 2].mean()
    saturation = hsv[:, :, 1].mean()
    
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]
    
    blue_sum = np.sum(blue_channel)
    green_sum = np.sum(green_channel)
    red_sum = np.sum(red_channel)
    
    total_sum = blue_sum + green_sum + red_sum
    
    blue_percentage = (blue_sum / total_sum) * 100
    green_percentage = (green_sum / total_sum) * 100
    red_percentage = (red_sum / total_sum) * 100

    return mean_brightness, saturation, blue_percentage, green_percentage, red_percentage