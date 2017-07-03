import numpy as np
import cv2
import mahotas

def reSize(image,width= None,height = None):
    if height is None:
        r = width / image.shape[1]
        dim = (width, int(image.shape[0] * r))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    if width is None:
        r = height / image.shape[0]
        dim = (int(image.shape[1] * r), height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def rotate(image, angle, scale):
    (h,w) = image.shape[:2]
    center = (w//2,h//2)
    M = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
    rotated = cv2.warpAffine(image,M,(w,h))
    return rotated

def translate(image, x, y):
    M = np.float32([1,0,x],[0,1,y])
    trans = cv2.warpAffine(image, M, (image.shape[1], image.shape[2]))
    return trans

def rectMasking(image,  mX, mY, center = None, sX = None, sY = None):
    mask = np.zeros(image.shape[:2], dtype="uint8")

    if center == "center":
        (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.rectangle(mask, (cX - mX, cY - mY), (cX + mX, cY + mY), 255, -1)
    else:
        cv2.rectangle(mask, (sX - mX, sY - mY), (sX + mX, sY + mY), 255, -1)

    masked = cv2.bitwise_and(image,image,mask=mask)
    return masked

def circMasking(image,center, radius):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius,255,-1)
    masked = cv2.bitwise_and(image,image,mask=mask)
    return masked

def cannyDetection(image, gaus, upperBound, lowerBound):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, gaus,0)
    canny = cv2.Canny(image,lowerBound, upperBound)
    return canny


def load_digits(datasetPath):
    #verilen pathi unsigned 8 bit numpy arrayine donusturur
    data = np.genfromtxt(datasetPath,delimiter=",",dtype="uint8")
    #0-9 arasi sayilari iceren kisim target degiskenine aktarilir
    target = data[:,0]
    #sayilarin pixelleri olan kisim data degiskenine aktarilir
    #pixel degerleri 0 ile 255 arasindadir
    data = data[:,1:].reshape(data.shape[0],28,28)

    return (data,target)
#deskew fonksiyonu yazim farkliliklarinda olusan hatalari gidermek icin kullanilir
def deskew(image, width):
    (h,w) = image.shape[:2]
    #resmin momentleri moments degiskenine aktarilir
    moments = cv2.moments(image)

    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([
        [1,skew,-0.5*w*skew],
        [0,1,0]
    ])
    #ilk parametre egrilestirilecek resmi ikincisi M yani hangi yone egrilestirilecegini belirten matris
    # sonuncusu resmin nasil deskewed yapilacagidir
    image = cv2.warpAffine(image,M,(w,h),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    image = reSize(image,width)

    return image


def center_extent(image,size):
    (eW,eH) = size

    if image.shape[1] > image.shape[0]:
        image = reSize(image,width=eW)
    else:
        image = reSize(image, height=eH)

    extent = np.zeros((eH,eW),dtype="uint8")
    offsetX= (eW - image.shape[1]) //2
    offsetY = (eH - image.shape[0]) //2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image

    CM = mahotas.center_of_mass(extent)
    (cY , cX) = np.round(CM).astype("int32")
    (dX, dY) = ((size[0]//2)-cX, (size[1]//2)-cY)
    M = np.float32([[1,0,dX],[0,1,dY]])
    extent = cv2.warpAffine(extent,M,size)

    return extent
