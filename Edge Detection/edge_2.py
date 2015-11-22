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

def matrix_convolution_row():

    # matriks untuk mendapatkan color value yang baru menggunakan sobel
    mc_row = np.asarray([-1,-2,-1,0,0,0,1,2,1])
    return mc_row

def matrix_convolution_col():

    # matriks untuk mendapatkan color value yang baru menggunakan sobel
    mc_col = np.asarray([-1,0,1,-2,0,2,-1,0,1])
    return mc_col

def edge_detection_sobel(imggrey,filter1,filter2):

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
    filter1 = matrix_convolution_row()
    filter2 = matrix_convolution_col()

    img_convolve = edge_detection_sobel(new_img,filter1,filter2)
    img_homogen = edge_detection_homogen(new_img)
    img_diff = edge_detection_diff(new_img)

    figure  = plt.figure( figsize=(15, 7) )
    figure.canvas.set_window_title( 'Image Convolve' )

    axes    = figure.add_subplot(121)
    axes.set_title('Original')
    axes.get_xaxis().set_visible( False )
    axes.get_yaxis().set_visible( False )
    axes.imshow( img, cmap='Greys_r' )

    axes    = figure.add_subplot(122)
    axes.set_title('Image Convolve')
    axes.get_xaxis().set_visible( False )
    axes.get_yaxis().set_visible( False )
    axes.imshow( img_convolve, cmap='Greys_r' )


    plt.show()





    # plt.imshow(img_convolve, cmap = 'Greys')
    # plt.show()
    # plt.imshow(img_homogen, cmap = 'Greys')
    # plt.show()
    # plt.imshow(img_diff, cmap = 'Greys')
    # plt.show()
    # number = random.randint(0,500)
    # misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\sbl'+str(number)+'.jpg',img_convolve)
    # misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\homog'+str(number)+'.jpg',img_homogen)
    # misc.imsave('E:\Dokudoku\Kuliah\S2 ITB\Pengenalan Pola\Edge Detection\edge_img\diff'+str(number)+'.jpg',img_diff)
