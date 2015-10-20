from scipy import misc
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import functions

example = misc.imread('D:\Codes\pengenalan pola\chaincode\plat\\1.jpg')
bw = np.zeros((example.shape[0], example.shape[1]))
objectsList = []

alphanumeric = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2',
'3', '4', '5', '6', '7', '8', '9']

# straighforward copy from wikibooks
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

def classify(plateCC, dictionaryCC, plateKB, dictionaryKB):
    plateNum = ""
    # iterate every feature extracted from test picture
    # plateCC = list of chain codes from test picture
    # plateKB = list of kode belok from test picture
    # dictionaryXX = list of features from training picture
    # for every index in plateCC, iterate this:
    for i in xrange(len(plateCC)):
        matchScore = [] # store total matching score
        matchScoreCC = [] # store matching score according to chain code
        matchScoreKB = [] # store matching score according to kode belok

        # compare every feature with feature in dictionary
        # for every index in dictionaryCC, iterate this:
        for j in xrange(len(dictionaryCC)):

            # for every chain code in dictionary, calculate its
            # levenshtein distance
            # save it to matchScoreCC
            distanceCC = levenshtein(plateCC[i], dictionaryCC[j])
            matchScoreCC.append(distanceCC)

            # do the same thing with kode belok
            # save it to matchScoreKB
            distanceKB = levenshtein(plateKB[i], dictionaryKB[j])
            matchScoreKB.append(distanceKB)

            # sum each score in matchScoreCC with its matchScoreKB counterpart
            score = matchScoreCC[j]+matchScoreKB[j]
            matchScore.append(score)

        chosen = matchScore.index(min(matchScore))
        plateNum += alphanumeric[chosen]
        print "distance: " + str(matchScore[chosen])
        print "chain code: " + str(dictionaryCC[chosen])
        print "kode belok: " + str(dictionaryKB[chosen])
    print plateNum
    # return plateNum

if __name__ == '__main__':

    # sementara
    bw = functions.getBW(example)
    plt.imshow(bw, cmap = 'Greys')
    plt.show()
    objectsList = functions.segmentation(bw)
    objectsList = np.asarray(objectsList)

    for n in xrange(len(objectsList)):
        plt.imshow(objectsList[n], cmap = 'Greys')
        plt.show()

    chainCodes = functions.getChainCode(objectsList)
    kodeBelok = functions.getKodeBelok(chainCodes)

    f = open('chaincode.txt', 'r')
    g = open('kode_belok.txt', 'r')
    dictionaryCC = np.loadtxt(f, dtype = str, delimiter='||')
    dictionaryKB = np.loadtxt(g, dtype = str, delimiter='||')

    plateCC = chainCodes
    plateKB = kodeBelok
    classify(plateCC, dictionaryCC, plateKB, dictionaryKB)
