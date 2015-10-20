from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import copy

img = misc.imread('../zhangsuen/D_arial.jpg')
#bw = np.zeros((img.shape[0],img.shape[1]))


bwim = (0.2989 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8) #grayscale

binary =  bwim < 128 #blackwhite


bw = np.argwhere(binary)[0] #return index dari array hasil dari operasi boolean/ index dimana bwim < 128 rubah jadi list biasa

b = (bw - (0,1)).tolist()
firstpix = bw.tolist()

alphanumeric = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0','1', '2',
'3', '4', '5', '6', '7', '8', '9']



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


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def classify(cc, dictionaryCC, belok, dictionaryKB):
    plateNum = ""
    # iterate every feature extracted from test picture
    # plateCC = list of chain codes from test picture
    # plateKB = list of kode belok from test picture
    # dictionaryXX = list of features from training picture
    # for every index in plateCC, iterate this:

    matchScore = [] # store total matching score
    matchScoreCC = [] # store matching score according to chain code
    matchScoreKB = [] # store matching score according to kode belok

        # compare every feature with feature in dictionary
        # for every index in dictionaryCC, iterate this:
    for j in xrange(len(dictionaryCC)):

        # for every chain code in dictionary, calculate its
        # levenshtein distance
        # save it to matchScoreCC
        distanceCC = levenshtein(cc, dictionaryCC[j])
        matchScoreCC.append(distanceCC)

        # do the same thing with kode belok
        # save it to matchScoreKB
        distanceKB = levenshtein(belok, dictionaryKB[j])
        matchScoreKB.append(distanceKB)

        # sum each score in matchScoreCC with its matchScoreKB counterpart
        score = matchScoreCC[j]+matchScoreKB[j]
        matchScore.append(score)

    chosen = matchScore.index(min(matchScore))
    plateNum += alphanumeric[chosen]
    #print "distance: " + str(matchScore[chosen])
    print "chain code: " + str(dictionaryCC[chosen])
    print "kode belok: " + str(dictionaryKB[chosen])
    print plateNum
    # return plateNum


# def startpixel(binary):
#     startrow = 0
#     while True:
#         scanline = binary[startrow,:]
#         if np.any(scanline):
#             startcol = np.argwhere(scanline)[0,0] #array ambil elemen ke 0  colomn
#             break
#         startrow += 1
#     return [startrow,startcol]

# def dfs(binary):
#     startpixel = startpixel(binary)
#     listpixelindex = [startpixel]
#     stack = [startpixel] #selama masih ada isinya
#     bound = False
#     while stack:
#         row, col = stack.pop() #ambil pixel yang dimasukin terakhir
#         if row == 0 or col == 0: bound = True
#         edges = np.argwhere(binary[row-1:row+2, col-1:col+2]) - [1, 1]
#         for edge in edges:
#             nextpixel = [row+edge[0], col+edge[1]]
#             if nextpixel not in objpix:
#                 stack += [nextpixel]
#                 listpixelindex += [nextpixel]
#
#     [binary.itemset((pix[0], pix[1]), False) for pix in objpix]
#     if bound: return [], binary
#     else: return listpixelindex, binary


def wr(chaincode,kdbelok):
    with open('chaincode_arial.txt','a') as f:
        f.write(chaincode)
    with open('kode_belok_arial.txt','a') as f:
        f.write(kdbelok)
#
# def rd(content):
#     with open('/Users/dhikanbiya/Dropbox/Kuliah/IF_semester_1/Pattern Recognation/IF5181/chaincode+kodebelok/kamus','r') as f:
#         raw = f.read().split('\n')



if __name__ == '__main__':
    f = open('chaincode_arial.txt', 'r')
    g = open('kode_belok_arial.txt', 'r')
    dictionaryCC = np.loadtxt(f, dtype = str, delimiter='||')
    dictionaryKB = np.loadtxt(g, dtype = str, delimiter='||')
    cc = getChaincode()
    belok = KodeBelok(cc)
    classify(cc, dictionaryCC, belok, dictionaryKB)
    # print 'chaincode = ', cc
    # print 'kode belok = ', belok

    # chaincode = '||'+str(cc)
    # kdbelok = '||'+str(belok)
    # wr(chaincode,kdbelok)
