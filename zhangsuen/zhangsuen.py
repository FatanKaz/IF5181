__author__ = 'Fatan Kasyidi'
from scipy import misc
import numpy as np
import copy
import matplotlib.pyplot as plt

img = misc.imread('B_comic.jpg')
bw = np.zeros((img.shape[0], img.shape[1]))
def getBW():
    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            if(np.sum(img[row][col]))/3 > 128:
                bw[row][col] = 0
            else:
                bw[row][col] = 1

def zhangSuen(obj):
    # plt.imshow(obj, cmap = 'Greys')
    # plt.show()
    print obj.shape
    erase = [0]
    while erase:
        erase = []
        for row in range(1, obj.shape[0]-1):
            for col in xrange(1, obj.shape[1]-1):
                neighbors = [obj[row-1][col], obj[row-1][col+1], obj[row][col+1],
                obj[row+1][col+1], obj[row+1][col], obj[row+1][col-1],
                obj[row][col-1], obj[row-1][col-1]]

                p2, p3, p4, p5, p6, p7, p8, p9 = neighbors
                black = sum(neighbors)
                #transition
                transition = 0

                for p in xrange(len(neighbors)):
                    if neighbors[p - 1] < neighbors[p]:
                        transition += 1

                if (obj[row][col] == 1 and
                    2 <= black <= 6 and
                    transition == 1 and
                    p2*p4*p6 == 0 and
                    p4*p6*p8 == 0):
                    erase.append((row,col))
        for row, col in erase: obj[row][col] = 0
        # plt.imshow(obj, cmap = 'Greys')
        # plt.show()

        erase = []
        for row in xrange(1, obj.shape[0]-1):
            for col in xrange(1, obj.shape[1]-1):
                neighbors = [obj[row-1][col], obj[row-1][col+1], obj[row][col+1],
                obj[row+1][col+1], obj[row+1][col], obj[row+1][col-1],
                obj[row][col-1], obj[row-1][col-1]]

                p2, p3, p4, p5, p6, p7, p8, p9 = neighbors
                black = sum(neighbors)
                #transition
                transition = 0

                for p in xrange(len(neighbors)):
                    if neighbors[p - 1] < neighbors[p]:
                        transition += 1

                if (obj[row][col] == 1 and
                    2 <= black <= 6 and
                    transition == 1 and
                    p2*p4*p8 == 0 and
                    p2*p6*p8 == 0):
                    erase.append((row,col))
        for row, col in erase: obj[row][col] = 0
        # plt.imshow(obj, cmap = 'Greys')
        # plt.show()

    plt.imshow(obj, cmap = 'Greys')
    plt.show()
    return obj

if __name__ == '__main__':
   getBW()
   zhangSuen(bw)