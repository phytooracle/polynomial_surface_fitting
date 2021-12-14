#!/usr/bin/env python3
"""
Author : Moshe Steyn and Travis Simmons
Date   : today
Purpose: Rock the Casbah
"""
# Sample deployment

# singularity build test.simg docker:phytooracle/polynomial_surface_fitting_s10
# singularity run test.simg -i ./Wintercrop_3/


import argparse
import os
import sys
import random
import glob
import math
import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering
import scipy.optimize as optimize
from scipy import asarray as ar, exp, sqrt
from scipy.optimize import curve_fit
from sympy import Symbol, Derivative
import sympy as sym
from sympy.utilities.lambdify import lambdify


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p',
                        '--plant_path',
                        help='plant_path',
                        metavar='str',
                        type=str)

    parser.add_argument('-c',
                        '--crop',
                        action = 'store_true')

    parser.add_argument('-po',
                        '--pointcloud_outdir',
                        help='Output directory for pointclouds',
                        default = 'segmentation_pointclouds',
                        metavar='str',
                        type=str)
                        

    parser.add_argument('-fo',
                        '--figures_outdir',
                        help='Output directory for figures',
                        default = 'plant_reports',
                        metavar='str',
                        type=str)

    parser.add_argument('-pf',
                        '--pointcloud_filename',
                        help='Pointcloud Output filename',
                        default = 'poly_crop',
                        metavar='str',
                        type=str)

    parser.add_argument('-d',
                        '--deg_of_poly',
                        help='Degree of polynomial to be fit',
                        default = 6,
                        type = int)

    parser.add_argument('-n',
                        '--meshgrid_number',
                        help='number controlling density of the meshgrid',
                        default = 100,
                        type = int)

    parser.add_argument('-t',
                        '--thresh',
                        help='clustering threshold',
                        default = 0.1,
                        type = float)

    parser.add_argument('-ct',
                        '--crop_thresh',
                        help='Size of final crop as a percentage of the whole are of the original pcd divided by two.',
                        default = 0.25,
                        type = float)



    return parser.parse_args()



def generate_before_after_plot(pcd_arr,new_pcd_arr,full_jpg_outpath):
    fig = plt.figure(figsize=(20, 15))
    ax0 = fig.add_subplot(121, projection='3d')
    ax0.scatter(pcd_arr[:,0], pcd_arr[:,1], pcd_arr[:,2], cmap=cm.autumn, c=pcd_arr[:,2], marker='.', alpha=0.4,s=0.5)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(new_pcd_arr[:,0], new_pcd_arr[:,1], new_pcd_arr[:,2], cmap=cm.autumn, c=new_pcd_arr[:,2], marker='.', alpha=0.4,s=0.5)
    plt.savefig(full_jpg_outpath, dpi = 80, format = 'jpg')
    plt.close(fig)

def generate_fitting_plot(predict_x0,predict_x1,y_new,sx20,sx21,z,closest_center,full_jpg_outpath):
    fig = plt.figure(figsize=(20, 15))
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(predict_x0, predict_x1, y_new, rstride=1, cstride=1, cmap=cm.Blues, alpha=0.4)
    ax1.scatter(sx20, sx21, z, c='b', marker='o',alpha=1)
    fig.colorbar(surf, ax=ax1)
    
    ax2 = fig.add_subplot(122)
    cs = ax2.contourf(predict_x0, predict_x1, y_new.reshape(predict_x0.shape),levels=20)
    ax2.contour(cs, colors='k')
    fig.colorbar(cs, ax=ax2)
    ax2.scatter(sx20,sx21,c='r',marker='.')
    ax2.scatter(closest_center[0],closest_center[1],c='black',marker='X',s=100)

    plt.savefig(full_jpg_outpath, dpi = 80, format = 'jpg')
    plt.close(fig)

# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    # variables
    args = get_args()

    # hyperparameters
    deg_of_poly = args.deg_of_poly
    N = args.meshgrid_number
    tresh = args.thresh
    crop_thresh = args.crop_thresh
    crop = args.crop

    # Inputs
    plant_path = args.plant_path
    pcd_path = os.path.join(plant_path, 'combined_multiway_registered.ply')
    plant_name = os.path.basename(plant_path)
    print(plant_name)



    # Outputs
    pointcloud_outdir = args.pointcloud_outdir
    pointcloud_filename = args.pointcloud_filename
    plant_pointcloud_outdir = os.path.join(pointcloud_outdir, plant_name)
    full_pointcloud_outpath = os.path.join(plant_pointcloud_outdir, pointcloud_filename + '.ply')

    figures_outdir = args.figures_outdir
    plant_figures_outdir = os.path.join(figures_outdir, plant_name)

    if crop:
        extension = 'poly_crop'

    else:
        extension = 'poly_n'


    full_csv_outpath = os.path.join(plant_figures_outdir, extension + '.csv')
    

    full_jpg_outpath = os.path.join(plant_figures_outdir, extension + '-before_after.jpg')
    full_fitting_jpg_outpath = os.path.join(plant_figures_outdir, extension + '-fitting.jpg')

    # prepping output directory structure
    if not os.path.exists(pointcloud_outdir):
        os.mkdir(pointcloud_outdir)

    if not os.path.exists(plant_pointcloud_outdir):
        os.mkdir(plant_pointcloud_outdir)

    if not os.path.exists(figures_outdir):
        os.mkdir(figures_outdir)

    if not os.path.exists(plant_figures_outdir):
        os.mkdir(plant_figures_outdir)


    # preparing point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_arr = np.asarray(pcd.points)
    xmin = pcd_arr[:,0].min()
    ymin = pcd_arr[:,1].min()
    movement = [xmin, ymin, 0]
    pcd_arr = pcd_arr - movement
    X = pcd_arr[:,0:2]
    Y = pcd_arr[:,-1]
    datapoints = pcd_arr

    # Fitting model
    poly = PolynomialFeatures(degree=deg_of_poly)
    X_ = poly.fit_transform(X)
    clf = linear_model.LinearRegression()
    clf.fit(X_, Y)

    xmax = pcd_arr[:,0].max()
    ymax = pcd_arr[:,1].max()

    predict_x0, predict_x1 = np.meshgrid(np.linspace(0, xmax, N),
                                        np.linspace(0, ymax, N))

    predict_x = np.concatenate((predict_x0.reshape(-1, 1), 
                                predict_x1.reshape(-1, 1)), 
                            axis=1)
    predict_x_ = poly.fit_transform(predict_x)
    predict_y = clf.predict(predict_x_)

    coef = clf.coef_
    degs = poly.get_feature_names()

    # Converting to symbolic function and taking derivative
    x0= Symbol('x0')
    x1= Symbol('x1')

    function = 0

    for i,c in enumerate(coef):
        d = degs[i]
        function+=c*eval(d.replace(' ','*').replace('^','**'))

    partialderiv_x0= Derivative(function, x0)
    partialderiv_x1= Derivative(function, x1)

    f_1 = partialderiv_x0.doit()
    f_2 = partialderiv_x1.doit()

    secondderivative_x0= Derivative(f_1, x0)
    secondderivative_x1= Derivative(f_2, x1)
    secondderivative_x0x1= Derivative(f_1, x1)

    f_11 = secondderivative_x0.doit()
    f_22 = secondderivative_x1.doit()
    f_12 = secondderivative_x0x1.doit()

    # Converting the partial derivatives to lambda function for numpy calculations
    func = lambdify([x0,x1], function,'numpy') 
    func1 = lambdify([x0,x1], f_1,'numpy') 
    func2 = lambdify([x0,x1], f_2,'numpy') 
    func11 = lambdify([x0,x1], f_11,'numpy') 
    func22 = lambdify([x0,x1], f_22,'numpy') 
    func12 = lambdify([x0,x1], f_12,'numpy')

    y_new = func(predict_x0,predict_x1)
    y_new_1 = func1(predict_x0,predict_x1)
    y_new_2 = func2(predict_x0,predict_x1)
    y_new_11 = func11(predict_x0,predict_x1)
    y_new_22 = func22(predict_x0,predict_x1)
    y_new_12 = func12(predict_x0,predict_x1)

    # Finding the peaks using derivative 0.25test
    D = y_new_11*y_new_22-y_new_12**2
    sx20 = predict_x0[np.where((y_new_1<=tresh) & (y_new_1>=-tresh) & (y_new_2<=tresh) & (y_new_2>=-tresh) & (y_new_11<0) & (D>0))]
    sx21 = predict_x1[np.where((y_new_1<=tresh) & (y_new_1>=-tresh) & (y_new_2<=tresh) & (y_new_2>=-tresh) & (y_new_11<0) & (D>0))]
    z = func(sx20,sx21)

    # Finding clusters and centroids
    data = np.array([[x,sx21[i]] for i,x in enumerate(sx20)])
    clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=0.1).fit(data)
    labels = clustering.labels_

    pcd_center = np.mean(datapoints,axis=0)

    centers = []
    closest_center = None
    min_d = sys.maxsize

    for i in range(clustering.n_clusters_):
        points = data[labels==i]
        center = np.mean(points,axis=0)
        centers.append(center)
        d = math.sqrt((center[0]-pcd_center[0])**2+(center[1]-pcd_center[1])**2)

        if closest_center is None or d<min_d:
            closest_center = center
            min_d = d


    print(f":: Found {clustering.n_clusters_} clusters")
    print(f":: Closest to center is {closest_center}")
    # print(f":: Radius of cropping is {radius}")

    if crop:

        print(f":: Cropping pointcloud")
        # Crop around the center
        mins = np.min(datapoints,axis=0)
        maxs = np.max(datapoints,axis=0)

        width = maxs[0]-mins[0]
        length = maxs[1]-mins[1]

        bound_x = crop_thresh*width
        bound_y = crop_thresh*length

        new_pcd_arr = pcd_arr.copy()
        new_pcd_arr = new_pcd_arr[np.where((new_pcd_arr[:,0]>closest_center[0]-bound_x) & (new_pcd_arr[:,0]<closest_center[0]+bound_x) & (new_pcd_arr[:,1]>closest_center[1]-bound_y) & (new_pcd_arr[:,1]<closest_center[1]+bound_y))]

        # output file and csv
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(new_pcd_arr)
        o3d.io.write_point_cloud(full_pointcloud_outpath, final_pcd)

        # Create figures
        generate_before_after_plot(pcd_arr,new_pcd_arr,full_jpg_outpath)
        

        new_pcd_point_count = len(new_pcd_arr)

    else:
        new_pcd_point_count = 'NaN'

    generate_fitting_plot(predict_x0,predict_x1,y_new,sx20,sx21,z,closest_center,full_fitting_jpg_outpath)

    df = pd.DataFrame(columns=['plant_name','polynomial_degree', 'original_pcd_point_count', 'new_pcd_point_count', 'num_centers', 'center_cluster_cordinates'])

    original_pcd_point_count = len(pcd_arr)

    information = [plant_name, deg_of_poly, original_pcd_point_count, new_pcd_point_count, clustering.n_clusters_, closest_center ]
    df.loc[len(df)] = information

    df.to_csv(full_csv_outpath)






# --------------------------------------------------
if __name__ == '__main__':
    main()



















