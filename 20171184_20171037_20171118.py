from PIL import Image
from math import exp
import numpy as np
import cv2
import sys

slides_path=sys.argv[1]
if slides_path[-1]!='/':
	slides_path+='/'
frames_path=sys.argv[2]
if frames_path[-1]!='/':
	frames_path+='/'

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

from os import listdir
from os.path import isfile, join
frames = [f for f in listdir(frames_path) if isfile(join(frames_path, f))]
slides = [f for f in listdir(slides_path) if isfile(join(slides_path, f))]
frames.sort()

cache_val={}

def orb_matching(img1,img2,img1_name,img2_name):
	orb = cv2.ORB_create()
	try:
		des1=cache_val[img1_name]
	except:
		kp1, des1 = orb.detectAndCompute(img1,None)
		cache_val[img1_name]=des1
	try:
		des2=cache_val[img2_name]
	except:
		kp2, des2 = orb.detectAndCompute(img2,None)
		cache_val[img2_name]=des2

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# ratio test as per Lowe's paper

	count=0
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			count+=1
	return count

def corr(im1,im2):
	product=np.mean((im1-im1.mean())*(im2-im2.mean()))
	stds=im1.std()*im2.std()
	if stds==0:
		return 0
	else:
		return product/stds

def windowed_corr(im1,im2):
	r=im1.shape[0]
	c=im1.shape[0]
	w_size_r=50
	w_size_c=50
	corr_arr=np.zeros((r//w_size_r + 1,c//w_size_c + 1))
	for i in range(0,r,w_size_r):
		for j in range(0,c,w_size_c):
			corr_arr[i//100,j//100]=corr(im1[i:i+100,j:j+100],im2[i:i+100,j:j+100])
	return sum(sum(corr_arr))


results={}
images_frames={}
images_slides={}

for i in frames:
	images_frames[i]=cv2.imread(frames_path+i,0)
for j in slides:
	images_slides[j]=cv2.imread(slides_path+j,0)

for i in frames:
	m_val=0
	m_s=""
	d={}
	img1 = images_frames[i]
	for j in slides:
		img2 = images_slides[j]
		d[j]=orb_matching(img1,img2,i,j)
	arr=[]
	for w in sorted(d, key=d.get, reverse=True):
		arr.append(w)
	for j in arr:
		curr_val=windowed_corr(img1,images_slides[j])
		if curr_val>m_val:
			m_val=curr_val
			m_s=j

	results[i]=m_s
	print(i,m_s)

with open("20171184_20171037_20171118.txt","w") as f:
	for i in results:
		f.write(i+" "+results[i]+"\n")
