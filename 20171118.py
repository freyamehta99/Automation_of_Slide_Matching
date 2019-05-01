from PIL import Image
from math import exp
import numpy as np
import sys

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

from os import listdir
from os.path import isfile, join

fr = sys.argv[2]
sl = sys.argv[1]

frames = [fr+f for f in listdir(fr) if isfile(join(fr, f))]
slides = [sl+f for f in listdir(sl) if isfile(join(sl, f))]
frames.sort()

#print(frames,slides)
def foo(im1,im2):
	#print(im1.shape,im2.shape)
	def corr(im1,im2):
		product=np.mean((im1-im1.mean())*(im2-im2.mean()))
		stds=im1.std()*im2.std()
		if stds==0:
			return 0
		else:
			return product/stds

	#print(corr(im1,im2))

	#print(corr(im1[0:100,0:100,:],im2[0:100,0:100,:]))
	#print(corr(im1[540-50:540+50,699-50:699+50,:],im2[540-50:540+50,699-50:699+50,:]))

	r=im1.shape[0]
	c=im1.shape[0]
	w_size_r=50
	w_size_c=50
	corr_arr=np.zeros((r//w_size_r + 1,c//w_size_c + 1))
	for i in range(0,r,w_size_r):
		for j in range(0,c,w_size_c):
			corr_arr[i//100,j//100]=corr(im1[i:i+100,j:j+100,:],im2[i:i+100,j:j+100,:])
	#print(corr_arr)
	m_r,m_c = np.unravel_index(corr_arr.argmax(), corr_arr.shape)
	gauss_corr=corr_arr
	for i in range(corr_arr.shape[0]):
		for j in range(corr_arr.shape[1]):
			gauss_corr[i][j]=corr_arr[i,j]*exp(-((i-m_r)**2 + (j-m_c)**2)/20)
	return sum(sum(gauss_corr))

results={}

for i in frames:
	m_val=-1e8
	m_s=''
	for j in slides:
		curr_val=foo(load_image(i),load_image(j))
		# print(i,j,curr_val)
		if curr_val>m_val:
			m_val=curr_val
			m_s=j
	results[i]=m_s
	print(results[i], i)

with open('20171118.txt', 'w') as roll_num:
	roll_num.write('\nRESULTS\n\n')
	for i in results:
		roll_num.write(i)
		roll_num.write(results[i])
		roll_num.write('\n')


# print("\n\nRESULTS\n\n")
