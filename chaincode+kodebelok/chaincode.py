from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import copy

img = misc.imread('A_arial.jpg')
#bw = np.zeros((img.shape[0],img.shape[1]))


bwim = (0.2989 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8) #grayscale

binary =  bwim < 128 #blackwhite


bw = np.argwhere(binary)[0] #return index dari array hasil dari operasi boolean/ index dimana bwim < 128 rubah jadi list biasa

b = (bw - (0,1)).tolist()
firstpix = bw.tolist()

def getDirection(firstpix, b):
    dir = 0
    row = firstpix[0]
    col = firstpix[1]
    if b == [row, col+1]:
        dir = 0
    if b == [row-1, col+1]:
        dir = 1
    if b == [row-1, col]:
        dir = 2
    if b == [row-1, col-1]:
        dir = 3
    if b == [row, col-1]:
        dir = 4
    if b == [row+1, col-1]:
        dir = 5
    if b == [row+1, col]:
        dir = 6
    if b == [row+1, col+1]:
        dir = 7

    return dir

def getIndex(dir, firstpix):
    row = firstpix[0]
    col = firstpix[1]
    if dir == 0:
        grid = [row, col+1]
    if dir == 1:
        grid = [row-1, col+1]
    if dir == 2:
        grid = [row-1, col]
    if dir == 3:
        grid = [row-1, col-1]
    if dir == 4:
        grid = [row, col-1]
    if dir == 5:
        grid = [row+1, col-1]
    if dir == 6:
        grid = [row+1, col]
    if dir == 7:
        grid = [row+1, col+1]

    return grid

def getChaincode():
    chainCode = []
    curInd = firstpix
    backtrack = b
    flag = copy.copy(curInd)
    flagstat = False

    while flagstat == False:
        pixVal = binary[backtrack[0],backtrack[1]]
        direction = getDirection(curInd,backtrack)

        while pixVal != True:
            direction = (direction+1) % 8
            newpixel = getIndex(direction, curInd)
            pixVal = binary[newpixel[0],newpixel[1]]

            if pixVal == False:
                backtrack = newpixel
            else:
                curInd = newpixel

        chainCode.append(direction)
        # print direction
        if curInd == flag:
            flagstat = True
    strcc = ''.join(str(e) for e in chainCode)
    return strcc


def KodeBelok(code):
    belok = ""
    for i in range(0,len(code)-1):
        dir = int(code[i])
        tambah = (dir+4)%8
        next = int(code[i+1])
        if next == (dir+1)%8 or next == (dir+2)%8 or next == (dir+3)%8:
            belok = belok + '-'
        elif next == (tambah+1)%8 or next==(tambah+2)%8 or next==(tambah+3)%8:
            belok = belok + '+'
        else:
            continue
    return belok



if __name__ == '__main__':
    cc = getChaincode()
    belok = KodeBelok(cc)
    print 'Chaincode',cc
    print 'Kode Belok ', belok
