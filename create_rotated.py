import cv2
import sys
import os

def rotate(image_dir, angle):
	output_dir=angle+image_dir
	os.makedirs(output_dir)
	for fn in os.listdir(image_dir):
		image=cv2.imread(image_dir+"/"+fn)
		rows, cols, d = image.shape
		M = cv2.getRotationMatrix2D((cols/2,rows/2),int(angle),1)
		image = cv2.warpAffine(image,M,(cols,rows))
		cv2.imwrite(output_dir+"/"+fn, image)


if __name__ == "__main__":
	rotate(sys.argv[1], sys.argv[2])
