from skimage import feature

class HOG:
    def __init__(self, orientations, pixelsPerCell,
cellsPerBlock , normalize = False):
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalize = normalize

    def describe(self,image):
        #hata bu kisimda olabilir dikkat et
        hist = feature.hog(image,orientations = self.orientations,pixels_per_cell=self.pixelsPerCell,cells_per_block = self.cellsPerBlock
                           )
        return hist