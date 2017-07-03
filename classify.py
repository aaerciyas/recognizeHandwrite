from __future__ import print_function
from sklearn.externals import joblib
from hog import HOG
from f import cvision
import argparse
import mahotas
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
help = "path to where the model will be stored")
ap.add_argument("-i", "--image", required = True,
help = "path to the image file")
args = vars(ap.parse_args())

model = joblib.load(args["model"])

hog = HOG(orientations = 18, pixelsPerCell = (10, 10),
cellsPerBlock = (1, 1), normalize = True)

image = cv2.imread(args["image"])
image = cvision.reSize(image,width=350)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edged = cvision.cannyDetection(image, (5,5), 30,150)

(_, counts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#soldan saga siralar
counts = sorted([(c, cv2.boundingRect(c)[0]) for c in counts], key=lambda x: x[1])
for(c, _) in counts:
    #karelerin alanlarini hesaplamakta
    (x, y, w, h) = cv2.boundingRect(c)

    if w>=7 and h>=20:
        #region of interest
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        thresh = cv2.bitwise_not(thresh)

        thresh = cvision.deskew(thresh, 20)
        thresh = cvision.center_extent(thresh, (20, 20))

        cv2.imshow("thresh", thresh)

        hist= hog.describe(thresh)
        digit = model.predict(hist)[0]
        print("Number is probably: {}".format(digit))

        cv2.rectangle(image, (x,y),(x + w , y + h), (0, 255, 0), 1)
        cv2.putText(image,str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0 , 255, 0), 2)
        cv2.imshow("image",image)
        cv2.waitKey(0)
