import cv2
import numpy as np
import os
import glob

def histequ(gray, nlevels=256):
    # Compute histogram
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    print ("histogram: ", histogram)

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    # print ("uniform hist: ", uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity
    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            uniform_gray[i,j] = uniform_hist[gray[i,j]]

    return uniform_gray

if __name__ == "__main__":
    Orignal_DataPath = "../Orignal_data"
    Histequa_DataPath="../Histequ_data"

    for indir in os.listdir(Orignal_DataPath):
        print("indir is %d" % int(indir))
        imgs_data = glob.glob(os.path.join(Orignal_DataPath, indir) + '/*' + '.png')
        for l in range(1, len(imgs_data) + 1):
            Histequa_SavePath=Histequa_DataPath+'/'+str(indir)
            if not os.path.lexists(Histequa_SavePath):
                os.mkdir(Histequa_SavePath)  # 聚类之后的存储路径
#////////////////直方图均衡化//////////////
            img_path=Orignal_DataPath+'/'+ str(indir)+'/' + str(l) + ".png"
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            uniform_gray=histequ(gray)
            cv2.imwrite(Histequa_SavePath + '/' + str(l) + '.png', uniform_gray)

