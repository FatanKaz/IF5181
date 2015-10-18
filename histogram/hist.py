from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('penguins.jpg')
img = list(img.getdata())

# alamat = raw_input("pilih(tulis) gambar : ")
# gambar = misc.imread(alamat)
# tampil = Image.open(alamat)
# tampil.show()

hist_r = [0]*256
hist_g = [0]*256
hist_b = [0]*256

for pixel in img:
	hist_r[pixel[0]] += 1
	hist_g[pixel[1]] += 1
	hist_b[pixel[2]] += 1

x = range(len(hist_r))
plt.plot(x,hist_r,'r', x,hist_g,'g', x,hist_b,'b')
plt.show()