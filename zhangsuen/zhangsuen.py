from scipy import misc
import numpy as np
import copy
import matplotlib.pyplot as plt

img = misc.imread('A_arial.jpg')
bw = np.zeros((img.shape[0], img.shape[1]))

def getBW():
    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            if(np.sum(img[row][col]))/3 > 128:
                bw[row][col] = 0
            else:
                bw[row][col] = 1

def neighbours(x,y,image):
    i = image
    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
    return [i[y1][x], i[y1][x1], i[y][x1], i[y_1][x],
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]

def transitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum((n1, n2) == (0,1) for n1,n2 in zip(n, n[1:]))

def zhangsuen(bw):
    changing1 = changing2 = 1
    while changing1 or changing2:
        # step 1
        changing1 = []
        for row in range(1, len(bw)-1):
            for col in range(1, len(bw[row])-1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(row,col,bw)
                if(bw[row][col] == 1 and
                    P4 * P6 * P8 == 0 and
                    P2 * P4 * P6 == 0 and
                    transitions(n) == 1 and
                    2 <= sum(n) <= 6):
                    changing1.append((row,col))
        for col, row in changing1:
            bw[row][col] = 0
        # step 2
        changing2 = []
        for row in range(1, len(bw) - 1):
            for col in range(1, len(bw[row]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(row,col, bw)
                if (bw[row][col] == 1 and    # (Condition 0)
                    P2 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing2.append((row,col))
        for col, row in changing2:
            bw[row][col] = 0
    return bw

if __name__ == '__main__':
    getBW()
    # print bw[1][1]
    # image = zhangsuen(bw)
    # print image
    n = [1,2,3,4,5,6,7,8]
    p2,p3,p4,p5,p6,p7,p8,p9 = n
    print len(n)
    for i in xrange(len(n)):
        print i
    # print bw[143][145]
    # print len(bw)-1
    # print len(bw[1])-1
    # plt.imshow(img, cmap = 'Greys')
    # plt.show()
    # bone = zhangsuen(bw)
    # print bone
    # print bw.tolist()
