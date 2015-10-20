from scipy import misc
import numpy as np
import math
import copy
import chaincode
import matplotlib.pyplot as plt

example = misc.imread('A_arial.jpg')
bw = np.zeros((example.shape[0], example.shape[1]))
objectsList = []

# PREPROCESSING

def getBW(img):
    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            if (np.sum(img[row][col]))/3 > 128: # depends on the image file
                bw[row][col] = 0
            else:
                bw[row][col] = 1
    return bw

def findPixel(obj):
    bpixel = (0, 0)
    wpixel = (0, 0)
    for row in xrange(obj.shape[0]):
        for col in xrange(obj.shape[1]):
            if obj[row][col] == 1:
                bpixel = (row, col)
                wpixel = (row, col-1)
                return bpixel, wpixel

# segment image
def segmentation(img):
    # for now, use the same algorithm as getBorderElm, as dfs without thinning
    # would be costly
    # detail description on getChainCode
    imgcopy = np.copy(img)
    objectsList = []
    while np.any(imgcopy) == True:
        borderElm = []
        curInd, backtrack = findPixel(imgcopy)
        borderElm.append(curInd)
        flag = copy.copy(curInd)
        flagstat = False # mark starting pixel

        while flagstat == False:
            pixVal = imgcopy.item(backtrack)
            direction = chaincode.getDirection(curInd, backtrack)
            index = (0, 0)
            while pixVal != 1:
                direction = (direction+1) % 8
                newpixel = chaincode.getIndex(direction, curInd)
                pixVal = imgcopy.item(newpixel)
                if pixVal == 0:
                    backtrack = copy.copy(newpixel)
                else:
                    curInd = copy.copy(newpixel)
                    borderElm.append(curInd)

            if curInd == flag:
                flagstat = True

        # we will slice object and rescale it
        lowerBound = map(min, zip(*borderElm)) #bottom-right index
        upperBound = map(max, zip(*borderElm)) #top-left index
        temp = img[lowerBound[0]:upperBound[0]+1, lowerBound[1]:upperBound[1]+1]
        newobj = scaling(temp)
        #save rescaled image to objectsList
        objectsList.append(newobj.tolist())
        imgcopy[lowerBound[0]:upperBound[0]+1, lowerBound[1]:upperBound[1]+1] = 0

    # for now, lower and upper bounds are not saved since test picture is not
    # skewed and it only consists of capital letters, so 'coincidentally'
    # the computer reads it as human would read it. if features were to be
    # extracted from photo taken by user, chances are the photo will be skewed
    # a little, if not much, and therefore there's no guarantee that the program
    # will still read such photo the human way

    # for testing only
    plt.imshow(np.asarray(objectsList[0]), cmap = 'Greys')
    plt.show()

    return objectsList

# supposedly rescale an object to 40x40
# but rescale to 25x25 instead
# still not working the right way
def scaling(obj):
    newimg = np.zeros((42, 42))
    #this will copy rescaled image to a new array
    rrow = math.ceil(obj.shape[0]*1.0/40) #ratio of former:current row
    rcol = math.ceil(obj.shape[1]*1.0/40) #ratio of former:current col
    print obj.shape
    print rrow, rcol
    a = 1 # row index of rescaled image
    for i in xrange(obj.shape[0]):
        b = 1 # col index of rescaled image
        if i%(rrow) == 0:
            for j in xrange(obj.shape[1]):
                if j%(rcol) == 0:
                    newimg[a][b] = obj[i][j]
                    b +=1
            a += 1
    return newimg

# skeletonize object using zhang suen algorithm
def zhangSuen(obj):
    plt.imshow(obj, cmap = 'Greys')
    plt.show()
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

# FEATURE EXTRACTION
# CHAIN CODE

#get starting direction from neighboring current index
#param: current black pixel index, former white pixel index
#will be used later for backtracking
def getDirection(bpixel, wpixel):
    dir = 0
    row = bpixel[0]
    col = bpixel[1]
    if wpixel == (row, col+1):
        dir = 0
    if wpixel == (row-1, col+1):
        dir = 1
    if wpixel == (row-1, col):
        dir = 2
    if wpixel == (row-1, col-1):
        dir = 3
    if wpixel == (row, col-1):
        dir = 4
    if wpixel == (row+1, col-1):
        dir = 5
    if wpixel == (row+1, col):
        dir = 6
    if wpixel == (row+1, col+1):
        dir = 7

    return dir

#get index of moore neighbor
#param: rotating direction and current pixel index
def getIndex(dir, bpixel):
    row = bpixel[0]
    col = bpixel[1]
    if dir == 0:
        #grid = (row, col)
        grid = (row, col+1)
    if dir == 1:
        grid = (row-1, col+1)
    if dir == 2:
        grid = (row-1, col)
    if dir == 3:
        grid = (row-1, col-1)
    if dir == 4:
        grid = (row, col-1)
    if dir == 5:
        grid = (row+1, col-1)
    if dir == 6:
        grid = (row+1, col)
    if dir == 7:
        grid = (row+1, col+1)

    return grid

# get the chaincodes of all objects in the list (unskeletonized)
# will return a list of chain codes
def getChainCode(objectsList):
    #for every object in objectsList
    chainCodes = []

    for obj in objectsList:
        chainCode = []
        temp = obj # copy object so that object in list is not affected
        curInd, backtrack = findPixel(temp)
        # print "curInd, backtrack = " + str(curInd) + ", " + str(backtrack)
        flag = copy.copy(curInd)
        flagstat = False # mark starting pixel

        while flagstat == False: # while starting pixel is not encountered
            # point to backtrack pixel and store its pixel value
            # pix value should be 0 since backtrack pixel is white pixel
            pixVal = temp.item(backtrack)
            # get initial direction to backtrack pixel relative to current index
            # getDirection function has been declared above
            direction = getDirection(curInd, backtrack)
            index = (0, 0)
            # check all the neighbors of current index, starting from
            # direction
            # while pix value in that direction is not 1
            while pixVal != 1:
                # point to another direction
                # get the new pixel index in that direction
                # and get the value of that new pixel
                direction = (direction+1) % 8
                newpixel = getIndex(direction, curInd)
                pixVal = temp.item(newpixel)
                # if pix value is 0 then update backtrack pixel
                # else, update current pixel
                if pixVal == 0:
                    backtrack = copy.copy(newpixel)
                else:
                    curInd = copy.copy(newpixel)
            chainCode.append(direction)
            # print "curInd, backtrack = " + str(curInd) + ", " + str(backtrack)

            if curInd == flag:
                flagstat = True

        strcc = ''.join(str(e) for e in chainCode)
        chainCodes.append(strcc)

    return chainCodes

# KODE BELOK
# get kode belok of all chain codes in the list
# will return a list of kode belok
def getKodeBelok(chainCodes):
    beloks = []

    for code in chainCodes:
        belok = ""
        for i in range(0, len(code)-1):
            dir = int(code[i])
            cross = dir + 4
            next = int(code[i+1])
            if next == (dir+1)%8 or next == (dir+2)%8 or next == (dir+3)%8:
                belok = belok + '-'
            elif next == (cross+1)%8 or next == (cross+2)%8 or next == (cross+3)%8:
                belok = belok + '+'
            else:
                continue
        beloks.append(belok)
    return beloks

# SKELETON RECOGNITION
# extract feature like end and intersection (and supposedly hole) for each
# object
def skeletonRecognition(objectsList):
    #attemp to count the numbers of end(s) and intersection(s) of every object
    #in the objectsList
    listObjFeat = []

    for obj in objectsList:
        intersection = 0
        end = 0
        holes = 0
        stack = [0]
        stackcc = [0]
        zschainCode = ""
        visited = []
        listOfBlack = np.transpose(np.nonzero(obj))
        listOfBlack = tuple(map(tuple, listOfBlack))
        #find the index of the first pixel in the object
        #assign this to current index (curInd)
        curInd = findPixel(obj)[0]
        row = curInd[0]
        col = curInd[1]

        #for testing only
        count = 1
        intersections = []

        #test every pixel of obj
        while stack:
            print "iteration " + str(count)
            #remove the topmost element of stack
            stack.pop()
            stackcc.pop()

            #add visited pixel to list of visited pixel
            visited.append(curInd)
            print "current pixel index: " + str(curInd)

            #slice 3x3 array with curInd as the center pixel
            checkNeighbors = obj[row-1:row+2, col-1:col+2]

            # add black neighbor(s) to stack if it's not yet visited
            # ordered anti-clockwise according to freeman chain code
            # starting from zero; if center pixel is (row, col)
            # then direction 0 is (row-1, col+1)
            neighborInd = ((row, col+1), (row-1, col+1), (row-1, col),
            (row-1, col-1), (row, col-1), (row+1, col-1), (row+1, col),
            (row+1, col+1))

            for i in xrange(len(neighborInd)):
                if (neighborInd[i] in listOfBlack and
                    neighborInd[i] not in visited):
                    stack.append(neighborInd[i])
                    stackcc.append(i)
                elif neighborInd[i] in visited:
                    holes += 1

            print "current stack: " + str(stack)

            #count the amount of black neighbor(s)
            #if it's only one, then it's an end
            # -1 because center pixel is not counted
            blacks = np.argwhere(checkNeighbors)
            black = len(blacks) - 1
            if black == 1:
                end += 1

            #count transition(s) of the neighbors
            #if transition is more than 2, then it's an intersection
            transition = 0
            neighbors = [obj[row-1][col], obj[row-1][col+1], obj[row][col+1],
            obj[row+1][col+1], obj[row+1][col], obj[row+1][col-1],
            obj[row][col-1], obj[row-1][col-1]]

            p2, p3, p4, p5, p6, p7, p8, p9 = neighbors
            for p in xrange(len(neighbors)):
                if neighbors[p - 1] < neighbors[p]:
                    transition += 1
            if transition > 2:
                intersection += 1
                intersections.append(curInd)

            while stack and stack[-1] in visited:
                stack.pop()
                stackcc.pop()

            #update curInd value
            if len(stack) != 0:
                curInd = stack[-1]
                row = curInd[0]
                col = curInd[1]
                zschainCode = zschainCode + str(stackcc[-1])

            count += 1

        objectFeature = [end, intersection, holes, zschainCode]
        listObjFeat.append(objectFeature)
    print "list of intersections: " + str(intersections)

    return listObjFeat

if __name__ == '__main__':
    bw = getBW(example)
    obj = zhangSuen(bw)
    objectsList = [obj]

    listObjFeat = skeletonRecognition(objectsList)
    print listObjFeat

    # objecsList = [obj]
    # skeletonRecognition(objectsList)

    '''
    # bw = scaling(bw)
    objectsList = segmentation(bw)
    plt.imshow(bw, cmap = 'Greys')
    plt.show()
    # for n in xrange(len(objectsList)):
    #     plt.imshow(objectsList[n], cmap = 'Greys')
    #     plt.show()
    # objectsList.append(bw.tolist())

    # order the objects the dumb way
    orderedObj = [objectsList[1], objectsList[2], objectsList[3], objectsList[4],
    objectsList[5], objectsList[6], objectsList[7], objectsList[8],
    objectsList[9], objectsList[10], objectsList[0], objectsList[11],
    objectsList[12], objectsList[17], objectsList[26], objectsList[13],
    objectsList[19], objectsList[14], objectsList[20], objectsList[21],
    objectsList[22], objectsList[23], objectsList[15], objectsList[16],
    objectsList[24], objectsList[25], objectsList[18], objectsList[29],
    objectsList[30], objectsList[31], objectsList[27], objectsList[28],
    objectsList[32], objectsList[33], objectsList[34], objectsList[35]]

    objectsList = orderedObj

    # for n in xrange(len(objectsList)):
    #     plt.imshow(objectsList[n], cmap = 'Greys')
    #     plt.show()

    objectsList = np.asarray(objectsList)
    chainCodes = getChainCode(objectsList)
    kodeBelok = getKodeBelok(chainCodes)
    print chainCodes
    print kodeBelok

    # chainCodes = np.asarray(chainCodes)
    # chainCodes.tofile('D:\Codes\pengenalan pola\chaincode\chaincode.txt', sep='||',
    # format = "%s")
    # kodeBelok = np.asarray(kodeBelok)
    # kodeBelok.tofile('D:\Codes\pengenalan pola\chaincode\kode_belok.txt', sep='||',
    # format = "%s")
    '''
