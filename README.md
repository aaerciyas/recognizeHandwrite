# Handwriting Recognition 

This repo provides recognize handwriting digits used Python3-OpenCV cv2 and scikitlearn libraries. 

## Requirements:

[NumPy](https://www.scipy.org/scipylib/download.html)

[scikit-learn](http://scikit-learn.org/stable/install.html)

[mahotas](http://mahotas.readthedocs.io/en/latest/install.html)

[scikit-image](http://scikit-image.org/docs/dev/install.html)


### For train dataset:

<code>
python3 train.py --dataset data/digits.csv --model models/svm.cpickle 
</code>

### For classify the numbers in the image:

<code>
python3 classify.py --image images/numbers.png --model models/svm.cpickle 
</code>

