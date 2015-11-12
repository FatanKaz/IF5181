__author__ = 'Fatan Kasyidi'

from scipy import misc
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

img = misc.imread('IMG00273-20120308-2013.jpg')

def grayscale(obj):

    # get image grayscale
    return np.dot(obj[...,:3], [0.299, 0.587, 0.144])

def histogram_gc(img_gc):

    histogram_img_gc = []
    for row in xrange(0,img_gc.shape[0]-1):
        for col in xrange(0, img_gc.shape[1]-1):

            color = img_gc[row][col]
            histogram_img_gc[color] += 1

    return histogram_img_gc

def matrix_convolution_row_sobel():

    # matriks untuk mendapatkan color value yang baru menggunakan sobel
    mc_row = np.asarray([-1,-2,-1,0,0,0,1,2,1])
    return mc_row

def matrix_convolution_col_sobel():

    # matriks untuk mendapatkan color value yang baru menggunakan sobel
    mc_col = np.asarray([-1,0,1,-2,0,2,-1,0,1])
    return mc_col

def matrix_convolution_row_prewitt():

    # matriks untuk mendapatkan color value yang baru menggunakan prewitt
    mc_row = np.asarray([-1,-1,-1,0,0,0,1,1,1])
    return mc_row

def matrix_convolution_col_prewitt():

    # matriks untuk mendapatkan color value yang baru menggunakan prewitt
    mc_col = np.asarray([-1,0,1,-1,0,1,-1,0,1])
    return mc_col

def matrix_convolution_kirsch_1():
    # matriks kirsch
    mc_1 = np.asarray([5,5,5,-3,0,-3,-3,-3,-3])
    return mc_1

def matrix_convolution_kirsch_2():
    # matriks kirsch
    mc_2 = np.asarray([5,5,-3,5,0,-3,-3,-3,-3])
    return mc_2

def matrix_convolution_kirsch_3():
    # matriks kirsch
    mc_3 = np.asarray([5,-3,-3,5,0,-3,5,-3,-3])
    return mc_3

def matrix_convolution_kirsch_4():
    # matriks kirsch
    mc_4 = np.asarray([-3,-3,-3,5,0,-3,5,5,-3])
    return mc_4

def matrix_convolution_kirsch_5():
    # matriks kirsch
    mc_5 = np.asarray([-3,-3,-3,-3,0,-3,5,5,5])
    return mc_5

def matrix_convolution_kirsch_6():
    # matriks kirsch
    mc_6 = np.asarray([-3,-3,-3,-3,0,5,-3,5,5])
    return mc_6

def matrix_convolution_kirsch_7():
    # matriks kirsch
    mc_7 = np.asarray([-3,-3,5,-3,0,5,-3,-3,5])
    return mc_7

def matrix_convolution_kirsch_8():
    # matriks kirsch
    mc_8 = np.asarray([-3,5,5,-3,0,5,-3,-3,-3])
    return mc_8

def matrix_convolution_prewitt_1():
    # matriks prewitt
    mc_1 = np.asarray([1,1,1,0,0,0,-1,-1,-1])
    return mc_1

def matrix_convolution_prewitt_2():
    # matriks prewitt
    mc_2 = np.asarray([0,1,1,-1,0,1,-1,-1,0])
    return mc_2

def matrix_convolution_prewitt_3():
    # matriks prewitt
    mc_3 = np.asarray([-1,0,1,-1,0,1,-1,0,1])
    return mc_3

def matrix_convolution_prewitt_4():
    # matriks prewitt
    mc_4 = np.asarray([-1,-1,0,-1,0,1,0,1,1])
    return mc_4

def matrix_convolution_prewitt_5():
    # matriks prewitt
    mc_5 = np.asarray([-1,-1,-1,0,0,0,1,1,1])
    return mc_5

def matrix_convolution_prewitt_6():
    # matriks prewitt
    mc_6 = np.asarray([0,-1,-1,1,0,-1,1,1,0])
    return mc_6

def matrix_convolution_prewitt_7():
    # matriks prewitt
    mc_7 = np.asarray([1,0,-1,1,0,-1,1,0,-1])
    return mc_7

def matrix_convolution_prewitt_8():
    # matriks prewitt
    mc_8 = np.asarray([1,1,0,1,0,-1,0,-1,-1])
    return mc_8

def edge_detection(imggrey,filter1,filter2):

    global sc1, sc2
    arr_sobel = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1,imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):
            slice = imggrey[row-1:row+2, col-1:col+2]
            rslice = slice.reshape(9)

            c1 = [rslice[0]*filter1[0],rslice[1]*filter1[1],
                  rslice[2]*filter1[2],rslice[3]*filter1[3],
                  rslice[4]*filter1[4],rslice[5]*filter1[5],
                  rslice[6]*filter1[6],rslice[7]*filter1[7],
                  rslice[8]*filter1[8]]

            c2 = [rslice[0]*filter2[0],rslice[1]*filter2[1],
                  rslice[2]*filter2[2],rslice[3]*filter2[3],
                  rslice[4]*filter2[4],rslice[5]*filter2[5],
                  rslice[6]*filter2[6],rslice[7]*filter2[7],
                  rslice[8]*filter2[8]]

            sc1 = sum(c1)
            sc2 = sum(c2)

            arr_sobel[row][col] = abs(sc1) + abs(sc2)

    # print arr_sobel

    r_arr_sobel = arr_sobel.reshape(arr_sobel.shape[0]*arr_sobel.shape[1])

    min_number = min(r_arr_sobel)
    max_number = max(r_arr_sobel)
    # print max_number
    # print min_number

    normal = ((r_arr_sobel - min_number)*255)/(max_number - min_number)
    normal = normal.reshape(imggrey.shape[0],imggrey.shape[1])

    return normal

def edge_detection_d2(imggrey,filter1,filter2,filter3,filter4,filter5,filter6,filter7,filter8):
    
    arr_kirsch = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1,imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):
            slice = imggrey[row-1:row+2, col-1:col+2]
            rslice = slice.reshape(9)

            c1 = [rslice[0]*filter1[0],rslice[1]*filter1[1],
                  rslice[2]*filter1[2],rslice[3]*filter1[3],
                  rslice[4]*filter1[4],rslice[5]*filter1[5],
                  rslice[6]*filter1[6],rslice[7]*filter1[7],
                  rslice[8]*filter1[8]]

            c2 = [rslice[0]*filter2[0],rslice[1]*filter2[1],
                  rslice[2]*filter2[2],rslice[3]*filter2[3],
                  rslice[4]*filter2[4],rslice[5]*filter2[5],
                  rslice[6]*filter2[6],rslice[7]*filter2[7],
                  rslice[8]*filter2[8]]
           
            c3 = [rslice[0]*filter3[0],rslice[1]*filter3[1],
                  rslice[2]*filter3[2],rslice[3]*filter3[3],
                  rslice[4]*filter3[4],rslice[5]*filter3[5],
                  rslice[6]*filter3[6],rslice[7]*filter3[7],
                  rslice[8]*filter3[8]]
            
            c4 = [rslice[0]*filter4[0],rslice[1]*filter4[1],
                  rslice[2]*filter4[2],rslice[3]*filter4[3],
                  rslice[4]*filter4[4],rslice[5]*filter4[5],
                  rslice[6]*filter4[6],rslice[7]*filter4[7],
                  rslice[8]*filter4[8]]
           
            c5 = [rslice[0]*filter5[0],rslice[1]*filter5[1],
                  rslice[2]*filter5[2],rslice[3]*filter5[3],
                  rslice[4]*filter5[4],rslice[5]*filter5[5],
                  rslice[6]*filter5[6],rslice[7]*filter5[7],
                  rslice[8]*filter5[8]]
           
            c6 = [rslice[0]*filter6[0],rslice[1]*filter6[1],
                  rslice[2]*filter6[2],rslice[3]*filter6[3],
                  rslice[4]*filter6[4],rslice[5]*filter6[5],
                  rslice[6]*filter6[6],rslice[7]*filter6[7],
                  rslice[8]*filter6[8]]
           
            c7 = [rslice[0]*filter7[0],rslice[1]*filter7[1],
                  rslice[2]*filter7[2],rslice[3]*filter7[3],
                  rslice[4]*filter7[4],rslice[5]*filter7[5],
                  rslice[6]*filter7[6],rslice[7]*filter7[7],
                  rslice[8]*filter7[8]]
           
            c8 = [rslice[0]*filter8[0],rslice[1]*filter8[1],
                  rslice[2]*filter8[2],rslice[3]*filter8[3],
                  rslice[4]*filter8[4],rslice[5]*filter8[5],
                  rslice[6]*filter8[6],rslice[7]*filter8[7],
                  rslice[8]*filter8[8]]

            sc1 = sum(c1)
            sc2 = sum(c2)
            sc3 = sum(c3)
            sc4 = sum(c4)
            sc5 = sum(c5)
            sc6 = sum(c6)
            sc7 = sum(c7)
            sc8 = sum(c8)

            arr_kirsch[row][col] = abs(sc1) + abs(sc2) + abs(sc3) + abs(sc4) + abs(sc5) + abs(sc6) + abs(sc7) + abs(sc8)

    r_arr_kirsch = arr_kirsch.reshape(arr_kirsch.shape[0]*arr_kirsch.shape[1])

    min_number = min(r_arr_kirsch)
    max_number = max(r_arr_kirsch)
    # print max_number
    # print min_number

    normal = ((r_arr_kirsch - min_number)*255)/(max_number - min_number)
    normal = normal.reshape(imggrey.shape[0],imggrey.shape[1])

    return normal

def edge_detection_homogen(imggrey):

    homogen = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1, imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):

            value = [abs(imggrey[row][col]-imggrey[row-1][col-1]),abs(imggrey[row][col]-imggrey[row-1][col]),
                     abs(imggrey[row][col]-imggrey[row-1][col+1]),abs(imggrey[row][col]-imggrey[row][col-1]),
                     abs(imggrey[row][col]-imggrey[row][col+1]),abs(imggrey[row][col]-imggrey[row+1][col-1]),
                     abs(imggrey[row][col]-imggrey[row+1][col]),abs(imggrey[row][col]-imggrey[row+1][col+1])]

            max_number = max(value)
            homogen[row][col] = max_number

    return homogen

def edge_detection_diff(imggrey):

    diff = np.zeros((imggrey.shape[0],imggrey.shape[1]))
    for row in range(1, imggrey.shape[0]-1):
        for col in range(1, imggrey.shape[1]-1):
            value = [abs(imggrey[row-1][col-1]-imggrey[row+1][col+1]),abs(imggrey[row-1][col]-imggrey[row+1][col]),
                     abs(imggrey[row-1][col+1]-imggrey[row+1][col-1]),abs(imggrey[row][col-1]-imggrey[row][col+1])]

            max_number = max(value)
            diff[row][col] = max_number

    return diff
if __name__ == '__main__':

    # print img.shape[0]
    # print img.shape[1]

    new_img = grayscale(img)

    # sobel and prewitt filter 2 kernels
    # filter1 = matrix_convolution_row_sobel()
    # filter2 = matrix_convolution_col_sobel()
    # filter_prewitt_1 = matrix_convolution_col_prewitt()
    # filter_prewitt_2 = matrix_convolution_col_prewitt()

    # kirsch filter 8 kernels
    filter_kirsch_1 = matrix_convolution_kirsch_1()
    filter_kirsch_2 = matrix_convolution_kirsch_2()
    filter_kirsch_3 = matrix_convolution_kirsch_3()
    filter_kirsch_4 = matrix_convolution_kirsch_4()
    filter_kirsch_5 = matrix_convolution_kirsch_5()
    filter_kirsch_6 = matrix_convolution_kirsch_6()
    filter_kirsch_7 = matrix_convolution_kirsch_7()
    filter_kirsch_8 = matrix_convolution_kirsch_8()

     # prewitt filter 8 kernels
    filter_prewitt_1 = matrix_convolution_prewitt_1()
    filter_prewitt_2 = matrix_convolution_prewitt_2()
    filter_prewitt_3 = matrix_convolution_prewitt_3()
    filter_prewitt_4 = matrix_convolution_prewitt_4()
    filter_prewitt_5 = matrix_convolution_prewitt_5()
    filter_prewitt_6 = matrix_convolution_prewitt_6()
    filter_prewitt_7 = matrix_convolution_prewitt_7()
    filter_prewitt_8 = matrix_convolution_prewitt_8()

    # img_convolve = edge_detection(new_img,filter1,filter2)
    # img_prewitt = edge_detection(new_img,filter_prewitt_1,filter_prewitt_2)
    # img_homogen = edge_detection_homogen(new_img)
    # img_diff = edge_detection_diff(new_img)
    img_edge_d2_kirsch = edge_detection_d2(new_img,filter_kirsch_1,filter_kirsch_2,filter_kirsch_3,filter_kirsch_4,filter_kirsch_5,filter_kirsch_6,filter_kirsch_7,filter_kirsch_8)
    img_edge_d2_prewitt = edge_detection_d2(new_img,filter_prewitt_1,filter_prewitt_2,filter_prewitt_3,filter_prewitt_4,filter_prewitt_5,filter_prewitt_6,filter_prewitt_7,filter_prewitt_8)
    # plt.imshow(img_convolve, cmap = 'Greys')
    # plt.show()
    # plt.imshow(img_homogen, cmap = 'Greys')
    # plt.show()
    # plt.imshow(img_diff, cmap = 'Greys')
    # plt.show()
    number = random.randint(0,500)
    # misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\sbl'+str(number)+'.jpg',img_convolve)
    # misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\homog'+str(number)+'.jpg',img_homogen)
    # misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\diff'+str(number)+'.jpg',img_diff)
    # misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\prew'+str(number)+'.jpg',img_prewitt)
    misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\kirsch_d2'+str(number)+'.jpg',img_edge_d2_kirsch)
    misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\prewitt_d2'+str(number)+'.jpg',img_edge_d2_prewitt)