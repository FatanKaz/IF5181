import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
# read in the image


def skin_detection(img):
    # separate out r, g, b channels
    r = img[:,:,0].astype('f')
    g = img[:,:,1].astype('f')
    b = img[:,:,2].astype('f')
    # generate the three quantities

    alpha = 3*b*r**2/(r+b+g)**3
    beta =  (r+g+b)/(3*r) + (r-g)/(r+g+b)
    gamma = (r*b+g**2)/(g*b)
    # finally we apply the rules:
    # plt.imshow((alpha>0.1276)&(beta<=0.9498)&(gamma<=2.7775),cmap='gray')
    plt.imshow((alpha>0.1276)&(beta<=0.9498)&(gamma<=2.7775))
    plt.show()



def floodfill(matrix, x, y):
    #"hidden" stop clause - not reinvoking for "c" or "b", only for "a".
    if matrix[x][y] == "a":
        matrix[x][y] = "c"
        #recursively invoke flood fill on all surrounding cells:
        if x > 0:
            floodfill(matrix,x-1,y)
        if x < len(matrix[y]) - 1:
            floodfill(matrix,x+1,y)
        if y > 0:
            floodfill(matrix,x,y-1)
        if y < len(matrix) - 1:
            floodfill(matrix,x,y+1)

if __name__ == '__main__':
    img = mpimg.imread("avatar_1.jpg")
    skin_detection(img)
