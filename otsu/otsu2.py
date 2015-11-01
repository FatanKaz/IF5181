import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

img = misc.imread('ondel.jpg')
imgs = Image.open('ondel.jpg')

# import pdb;pdb.set_trace()
width, height = imgs.size


def histogram(img):

    hist = [0]*256
    gray = (0.2989 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8) #grayscale
    # plt.imshow(gray, cmap = plt.get_cmap('gray'))
    # plt.show()
    #
    # plt.hist(gray.flatten(), 256)
    # plt.show()
    return gray

def otsu():

    hist = histogram(img)
    sum_all = 0

    for t in xrange(256):
        sum_all += t*hist[t]


    sum_back, w_back, w_for, var_max, threshold = 0, 0, 0, 0, 0

    for t in xrange(256):
        w_back += hist_data[t]
        if w_back == 0:
            continue
        w_for = total - w_back
        if w_for == 0:
            break
        #calculate classes means
        sum_back  += t*hist_data[t]
        mean_back = sum_back/w_back
        mean_for = (sum_all - sum_back)/ w_for

        #calculate beetween class variance
        var_beetween = w_back * w_for * (mean_back - mean_for)**2
        if(var_beetween > var_max):
            var_max = var_beetween
            threshold = t
    return threshold




if __name__ == '__main__':
    otsu()
