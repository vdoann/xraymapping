#!/usr/bin/python

#Author: Vicky Doan-Nguyen

from PIL import Image
import os, fnmatch
import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt
import math
from datetime import datetime
from operator import itemgetter, attrgetter
 
startTime = datetime.now()
print 'Start Time: ', startTime
 
centerx = 610
centery = 676
radius = 130 #inner
#radius = 225 #outer
deltaradius = 6 #15
 
strudir = './'
filelist = os.listdir(strudir)
filelist = fnmatch.filter(filelist, 'SD15-31_map_x-*')
#filelist = fnmatch.filter(filelist, 'SD15-31_map_x-18.10*')
#filelist = fnmatch.filter(filelist, 'SD15-31_map_x-18.10z-3.55_th000_spot784_14sec_SAXS')
#filelist = ['SD15-31_map_x-18.10z-3.55_th000_spot784_14sec_SAXS']
#filelist = filelist[:5]
 
datalist = []
pts = []
xcoords=[]
zcoords=[]
alldata=[]
alldatafile = []
allpts = []
xcoordspts = []
ycoordspts = []
bins = []
     
def main():
 
    for file in filelist:
        print file
        x = float(file.split('x-')[1].split('z-')[0])
        z = float(file.split('z-')[1].split('_')[0])
        spot = int(file.split('_spot')[1].split('_')[0])
        #im = Image.open(file).convert("I")
        #data = np.asarray(im)
        xcoords.append(x)
        zcoords.append(z)
 
        im = Image.open('%s' %file)
        im = im.convert('RGB')
        data = np.array(im)
 
        max_values, bin_num, binmod12 = calculate_max_index(data, origin=(centerx,centery))
        bins.append(binmod12)
        print 'bins'
        print bins
        #plot_original_image(data)
        #plot_polar_image(data, origin=(centerx,centery))
        #plot_directional_intensity(data, origin=None)
        #plt.show()
 
 
    sorted_data = sorted( zip(xcoords,zcoords,bins), key=itemgetter(0,1) )
    num_xcoords = len(list(set(xcoords)))
    num_zcoords = len(list(set(zcoords)))
    data_2d = []
    for i in range(num_xcoords):
        row = []
        for j in range(num_zcoords):
            if i*num_zcoords+j < len(sorted_data):
                row.append(sorted_data[i*num_zcoords+j][2])
            else:
                row.append(0)
        data_2d.append(row)
    print data_2d
 
    fig, ax = plt.subplots()
    p = ax.imshow(data_2d, cmap=plt.cm.gist_rainbow, interpolation='nearest',origin='lower')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('2')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(5)
    print 'x ticks', sorted(list(set(zcoords)))
    print 'y ticks', sorted(list(set(xcoords)))
    cbar = fig.colorbar(p,ticks=[0,1,2,3,4,5,6,7,8,9])
    cbar.ax.set_yticklabels(['0-5','6-11','12-17','18-23','24-29','30-35','36-41','42-47','48-53','54-60'])
    cbar.ax.tick_params(labelsize=40, width=5,size=25)
    cbar.outline.set_linewidth(5)
    plt.show()
 
def calculate_max_index(data, origin=(centerx,centery)):
    polar_grid, r, theta = reproject_image_into_polar(data, origin)
    r_range = [210,240]
    intensity = np.zeros(1024)
    for i in range(1024):
        for j in range(1024):
            if r[i] > r_range[0] and r[i] < r_range[1]:
                intensity[j] += polar_grid[i][j][0]
 
    array_intensity = intensity[370:540]
    max_intensity = max(array_intensity)
    for i,intensityvalue in enumerate(array_intensity):
        if intensityvalue == max_intensity:
            index_max = i
            binmod12 = math.floor((index_max*60/170)/6)
            print '%d is theta (deg) = %f' % (index_max, index_max*60/170)
    return max_intensity, index_max, binmod12
 
def plot_original_image(data):
    plt.imshow(data)
    circle1 = plt.Circle((centerx,centery),radius,color='b',fill=False)
    circle2 = plt.Circle((centerx,centery),radius-6,color='g',fill=False)
    circle3 = plt.Circle((centerx,centery),radius+6,color='g',fill=False)
 
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)
    fig.gca().add_artist(circle3)
    plt.show()
 
def plot_polar_image(data, origin=(centerx,centery)):
    polar_grid, r, theta = reproject_image_into_polar(data, origin)
 
    print len(r)
    print len(theta)
    r_range = [210,240]
    intensity = np.zeros(1024)
    for i in range(1024):
        for j in range(1024):
            if r[i] > r_range[0] and r[i] < r_range[1]:
                intensity[j] += polar_grid[i][j][0]
 
    array_intensity = intensity[370:540]
    #theta_range = [-math.pi/6,-math.pi/2]
    #if theta[i] > theta_range[0] and theta[i] < theta_range[1]:
    #array_intensity = np.asarray(intensity)
    max_intensity = max(array_intensity)
    for i,intensityvalue in enumerate(array_intensity):
        if intensityvalue == max_intensity:
            index_max=i
            print 'max intensity is %d at index of %d'  % (intensityvalue, i)            
             
    fig, ax = plt.subplots()
    ax.plot(theta, intensity)
    ax.plot(theta[370],intensity[370],marker='o',markerfacecolor='g')
    ax.plot(theta[540],intensity[540],marker='o',markerfacecolor='r')
    ax.plot(theta[index_max+370],max_intensity,marker='s',markerfacecolor='y')
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Intensity')
    #plt.show()
    # end new part #
     
    plt.figure()
    plt.imshow(polar_grid, extent=(theta.min(), theta.max(), r.max(), r.min()))
    plt.axis('auto')
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('Angle (radians)')
    plt.ylabel('R Coordinate (pixels)')
    plt.title('Image in Polar Coordinates')
     
def index_coords(data, origin=(centerx,centery)):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

#Refer to Justin Peel's post on polar coordinates. http://tinyurl.com/qes7zz2 

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta
 
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
 
def bin_by(x, y, nbins=30):
    """Bin x by y, given paired observations of x & y.
    Returns the binned "x" values and the left edges of the bins."""
    bins = np.linspace(y.min(), y.max(), nbins+1)
    # To avoid extra bin for the max value
    bins[-1] += 1
    indicies = np.digitize(y, bins)
    output = []
    for i in xrange(1, len(bins)):
        output.append(x[indicies==i])
 
    # Just return the left edges of the bins
    bins = bins[:-1]
    return output, bins
 
def reproject_image_into_polar(data, origin='centerx,centery'):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)
 
    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)
 
    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)
 
    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)
 
    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)               
    bands = []
    for band in data.T:
        zi = sp.ndimage.map_coordinates(band, coords, order=1)
        bands.append(zi.reshape((nx, ny)))
    output = np.dstack(bands)
     
    return output, r_i, theta_i
 
if __name__ == '__main__':
    main()
