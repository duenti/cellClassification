from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from skimage import io 
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle
import sys

# settings for LBP
radius = 3
n_points = 8 * radius
METHOD = 'uniform'
GRID_SIZE = 8
GRID_HALF = int(GRID_SIZE/2)

def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

def plot_grid_hist(x_c,y_c,image):
    if x_c != None and y_c != None:
        x = int(x_c)
        y = int(y_c)
        sub_grid = np.array(image[x-GRID_HALF:x+GRID_HALF,y-GRID_HALF:y+GRID_HALF])
        flatten_sub_grid = sub_grid.ravel()
        plt.figure(1)
        plt.hist(flatten_sub_grid, 40,alpha=0.3)
        plt.show()
        

#image = data.load('/home/neli/cellImage/cellClassification/imagens/Snap-16642.tif')
image = io.imread('/home/neli/cellImage/cellClassification/imagens/Snap-16642.tif',as_gray=True)
lbp = local_binary_pattern(image, n_points, radius, METHOD)
#io.imshow(lbp)

mutable_object = {} 
fig = plt.figure(0)
io.imshow(image)
def onclick(event):
    #Coordinates are getting inverted
    Y_coordinate = event.xdata
    X_coordinate = event.ydata
    plot_grid_hist(X_coordinate,Y_coordinate,lbp)
    mutable_object['click'] = X_coordinate

cid = fig.canvas.mpl_connect('button_press_event', onclick)
#lines, = plt.plot([1,2,3])
plt.show()
X_coordinate = mutable_object['click']
print(X_coordinate)

#plt.show()