from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from hog import HOG
from f import cvision
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True)
ap.add_argument("-m","--model",required= True)
args = vars(ap.parse_args())

(digits, target) = cvision.load_digits(args["dataset"])
data = []

hog = HOG(orientations = 18,pixelsPerCell = (10,10), cellsPerBlock=(1,1),normalize=True)

for image in digits:
    image = cvision.deskew(image,20)
    image = cvision.center_extent(image,(20,20))

    hist = hog.describe(image)
    data.append(hist)

model = LinearSVC(random_state = 42)
model.fit(data,target)

joblib.dump(model, "model.cpickle")
