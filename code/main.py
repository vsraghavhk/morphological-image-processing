from IPP import get_images
import numpy as np 
import cv2
import os

filepath = '../images/'
results = '../results/'
# file = 'Image_01.jpg'
# maskpath = '../masks/sv2_mask.jpg'

KSIZE = 7
# Works well for large images, while 5 works better for smaller images. 

# Boundary kernel has to be 7 or one size more than KSIZE

def main():
	images, image_name = get_images(filepath)

	for i in range(len(images)):
		eimg = images[i]
		file = image_name[i]
		resultfolder = results+file[:-4]+'/'
		if not os.path.exists(resultfolder):
			os.makedirs(resultfolder)

		# eimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		kernel = np.ones((KSIZE, KSIZE), np.uint8)
		
		### Custom erosion/dilation fns
		# # Closing
		# eimg = dilation(img, kernel)
		# eimg = erosion(eimg, kernel)
		
		# # Opening
		# eimg = erosion(eimg, kernel)
		# eimg = dilation(eimg, kernel)
		
		eimg = closing(eimg, kernel)
		eimg = opening(eimg, kernel)
		
		cv2.imwrite(resultfolder+'/Noextract_'+file, eimg)

		morphex = cv2.morphologyEx(eimg, cv2.MORPH_GRADIENT, kernel)
		cv2.imwrite(resultfolder+'/morphex_'+file, morphex)
		
		boundary = boundary_extraction(eimg, kernel)
		cv2.imwrite(resultfolder+'/boundary_'+file, boundary)
		
		morphex = cv2.Canny(morphex, 75, 85)
		cv2.imwrite(resultfolder+'/Canny-morphex_'+file, morphex)
	

def apply_mask(img, mask):
	for k in range(img.shape[2]):
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if mask[i][j][k] !=0:
					# print("h")
					img[i][j][k] = 255
	return img

def remove_mask(img, mask, og_img):
	for k in range(img.shape[2]):
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if mask[i][j][k] != 0:
					# print(mask[i][j][k])
					img[i][j][k] = og_img[i][j][k]
	return img

def opening(img, kernel):
	img = cv2.erode(img, kernel, iterations=1)
	img = cv2.dilate(img, kernel, iterations=1)
	return img

def closing(img, kernel):
	img = cv2.dilate(img, kernel, iterations=1)
	img = cv2.erode(img, kernel, iterations=1)
	return img

def erosion(img, kernel): # Min value
	ksize = kernel.shape[0]
	ix, iy, iz = img.shape
	
	for k in range(iz):
		for i in range(0, ix, ksize):
			if i+ksize > ix:
				break
			for j in range(0, iy, ksize):
				vmax = img[i:i+ksize, j:j+ksize, k].min()			
				if j+ksize > iy:
					break
				img[i:i+ksize, j:j+ksize, k] = vmax
	return img

def dilation(img, kernel): # Max value
	ksize = kernel.shape[0]
	ix, iy, iz = img.shape
	
	for k in range(iz):
		for i in range(0, ix, ksize):
			if i+ksize > ix:
				break
			for j in range(0, iy, ksize):
				vmax = img[i:i+ksize, j:j+ksize, k].max()			
				if j+ksize > iy:
					break
				img[i:i+ksize, j:j+ksize, k] = vmax
	return img

def boundary_extraction(img, kernel):
	eimg = cv2.erode(img, kernel)
	boundary = img - eimg 
	return boundary

if __name__ == '__main__':
    main()