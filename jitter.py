# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from functools import partial
from argparse import ArgumentParser
from skimage.transform import SimilarityTransform, warp, rotate
from data_loader import transform_data_frame

np.random.seed(1)

def jitter_image(image, sigma):
    xy = np.random.normal(0, sigma, 2)
    translation = SimilarityTransform(translation=(xy[0], xy[1]))
    return warp(image, translation)
        
def jitter_images(images, sigma):
    return list(map(partial(jitter_image, sigma=sigma), images)) 

def rotate_image(image, sigma):
    angle = np.random.normal(0, sigma)
    return rotate(image, angle)
    
def rotate_images(images, sigma):
    return list(map(partial(rotate_image, sigma=sigma), images))
    
def create_df(images, labels):
    images_reshaped = list(map(lambda image: image.reshape(784), images))
    images_with_labels = np.hstack((np.array(labels).reshape((len(images), 1)), images_reshaped))
    df = pd.DataFrame(data = images_with_labels)
    return df
    
def build_args_parser():
    parser = ArgumentParser(description="Data generator")
    parser.add_argument('--input', help='Path of the file with training data')
    parser.add_argument('--output', help='Path of the output file')
    parser.add_argument('--jitter', help='How many jittered images to generate - multiplier', type=int)    
    parser.add_argument('--sigma', help='Sigma of the jitter', type=float)
    parser.add_argument('--noise', help='How many noised images to generate - multiplier', type=int)
    parser.add_argument('--probability', help='Probability of changing single pixel', type=float)
    parser.add_argument('--rotate', help='How many rotated images to generate - multiplier', type=int)    
    parser.add_argument('--angle', help='Standard deviation of rotation in degrees', type=float)
    parser.add_argument('--preserve', help='Preserve input', type=bool)    
    return parser

def cli(cli_args):
    parser = build_args_parser()
    args = parser.parse_args(cli_args)
    if args.input and args.output:
        input_df = pd.read_csv(args.input)
        data = transform_data_frame(input_df)
        df_as_arrays = list(map(lambda row: np.array(row), input_df.values))
        if args.preserve:
            df = pd.DataFrame(data = df_as_arrays)
        else:
            df = pd.DataFrame()
        if args.jitter and args.sigma:
            for i in range(args.jitter):
                jittered_images = jitter_images(data.matrices_float, args.sigma)
                df_jittered = create_df(jittered_images, data.labels)
                df = df.append(df_jittered)
        if args.noise and args.probability:
            for i in range(args.noise):
                change_pixel = lambda pixel: 0 if np.random.random() < args.probability else pixel 
                change_pixel_vectorized = np.vectorize(change_pixel)    
                noised_images = list(map(change_pixel_vectorized, data.matrices_float))
                df_noised = create_df(noised_images, data.labels)  
                df = df.append(df_noised)
        if args.rotate and args.angle:
            for i in range(args.rotate):
                rotated_images = rotate_images(data.matrices_float, args.angle)
                df_rotated = create_df(rotated_images, data.labels)
                df = df.append(df_rotated)
        df.to_csv(args.output, index=False)
    else:
        parser.print_usage()
        
cli(sys.argv[1:])




    
