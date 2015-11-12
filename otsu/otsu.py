from numpy import *
from scipy import misc
import matplotlib.pyplot as plt


def otsu(hist, total):
    bins = len(hist)

    sum_of_total = 0 #total pixel
    for x in xrange(0, bins):
        sum_of_total += x * hist[x]

    weight_back = 0.0
    sum_back = 0.0
    variance = []

    for thres in xrange(0, bins):
        weight_back += hist[thres]
        if weight_back == 0:
            continue

        weight_fore = total - weight_back
        if weight_fore == 0:
            break

        sum_back += thres * hist[thres]
        mean_back = sum_back/ weight_back
        mean_fore = (sum_of_total - sum_back)/ weight_fore

        variance.append( weight_back * weight_fore * (mean_back - mean_fore)**2 )

    # find the threshold with maximum variances between classes
	otsu_thres = argmax(variance)
    return otsu_thres



def main(img):
    #convert grayscale
    grayscale = img.dot( [0.299, 0.587, 0.144])
    rows, cols = shape(grayscale)

    #create 256 histogram
    hist = histogram(grayscale, 256)[0]
    total = rows * cols
    thre = otsu(hist, total)

    figure  = plt.figure( figsize=(14, 6) )
    figure.canvas.set_window_title( 'Otsu thresholding' )

    axes    = figure.add_subplot(121)
    axes.set_title('Original')
    axes.get_xaxis().set_visible( False )
    axes.get_yaxis().set_visible( False )
    axes.imshow( img, cmap='Greys_r' )

    axes    = figure.add_subplot(122)
    axes.set_title('Otsu thresholding')
    axes.get_xaxis().set_visible( False )
    axes.get_yaxis().set_visible( False )
    axes.imshow( grayscale >= thre, cmap='Greys_r' )


    plt.show()



if __name__ == '__main__':
    img = misc.imread( 'plat1.jpg')
    main(img)
