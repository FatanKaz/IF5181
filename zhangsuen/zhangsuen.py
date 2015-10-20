__author__ = 'Fatan Kasyidi'
from scipy import misc
import numpy as np
import copy
import matplotlib.pyplot as plt

img = misc.imread('M_arial.jpg')
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
    # print obj.shape
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

    # plt.imshow(obj, cmap = 'Greys')
    # plt.show()
    return obj

def simpang(obj):
    intersection = 0
    obj = obj.astype(bool)
    for row in range(1, obj.shape[0]-1):
        for col in xrange(1, obj.shape[1]-1):
            n = [obj[row-1,col], obj[row-1,col+1], obj[row,col+1],
            obj[row+1,col+1], obj[row+1,col], obj[row+1,col-1],
            obj[row,col-1], obj[row-1,col-1],obj[row-1,col]]

            # p2, p3, p4, p5, p6, p7, p8, p9 = n
            # print np.diff(n), np.sum(np.diff(n))
            if np.sum(np.diff(n))/2 >= 3:
                intersection += 1
                # sum_intersection = np.sum(np.diff(n))

            #print n

    return intersection

def ujung(obj):
    endpoint = 0
    # blackindex = []
    obj = obj.astype(int)
    for row in range(1, obj.shape[0]-1):
        for col in xrange(1, obj.shape[1]-1):

            if obj[row,col] == 1:
                n = [obj[row-1][col], obj[row-1][col+1], obj[row][col+1],
                     obj[row+1][col+1], obj[row+1][col], obj[row+1][col-1],
                     obj[row][col-1], obj[row-1][col-1]]

                p2, p3, p4, p5, p6, p7, p8, p9 = n
                black = sum(n)
                # pri   nt np.diff(n), sum(np.diff(n))
                if black == 1:
                    endpoint += 1
                    # blackindex.append((row,col))
                    # print n, black
                    # if sum(np.diff(n)) == 2: # np.sum(np.diff(n))
                    #     endpoint += 1
                    # sum_intersection = np.sum(np.diff(n))
                    #print n

    # print blackindex
    return endpoint

def retrieve_data_training():
    with open('training arial (simpang,ujung,huruf).txt', 'r') as f:
        training = f.read().split('\n')

    # training_list = []
    # for i in len(training):
    #     training_list = training[i].split()

    training_split = [instance.split(',') for instance in training]
    training_int = [[int(instance[0]), int(instance[1]), instance[2]] for instance in training_split]

    # print training_list
    return training_int

def testing_data(training_int,u,s):

    some = 'tidak ketemu'
    ketemu = False
    i = 0
    while not ketemu:
        if training_int[i][0] == s and training_int[i][1] == u:
            some = training_int[i][2]
            # print training_int[i][0], training_int[i][1]
            ketemu = True
        i += 1
    return some
if __name__ == '__main__':
   getBW()
   tulang = zhangSuen(bw)
   # print tulang
   s = simpang(tulang)
   u = ujung(tulang)
   training_data = retrieve_data_training()

   print 'objek adalah :',testing_data(training_data,u,s)
   # print 'jumlah simpang :',s
   # print 'jumlah ujung :',u

   # print tulang[103]
   # print len(tulang)
   # plt.imshow(tulang, cmap = 'Greys')
   # plt.show()
