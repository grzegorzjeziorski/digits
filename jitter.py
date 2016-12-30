# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from functools import partial
from argparse import ArgumentParser
from skimage.transform import SimilarityTransform, warp
from data_loader import transform_data_frame

#data_frame = load_data_frame('~/Documents/digits/train.csv')

#data = transform_data_frame(data_frame)

np.random.seed(1)

def jitter_image(image, sigma):
    xy = np.random.normal(0, sigma, 2)
    translation = SimilarityTransform(translation=(xy[0], xy[1]))
    return warp(image, translation)
        
def jitter_images(images, sigma):
    return list(map(partial(jitter_image, sigma=sigma), images)) 
    
def create_df(images, labels):
    images_reshaped = list(map(lambda image: image.reshape(784), images))
    images_with_labels = np.hstack((np.array(labels).reshape((len(images), 1)), images_reshaped))
    df = pd.DataFrame(data = images_with_labels)
    return df
    
#jittered_images = jitter_images(data.matrices_float)
#df_jittered = create_df(jittered_images, data.labels)

#probability = 0.025    
#change_pixel = lambda pixel: 0 if np.random.random() < probability else pixel 
#change_pixel_vectorized = np.vectorize(change_pixel)    
#noised_images = list(map(change_pixel_vectorized, data.matrices_float))
#df_noised = create_df(noised_images, data.labels)

#df_as_arrays = list(map(lambda row: np.array(row), data_frame.values))
#df = pd.DataFrame(data = df_as_arrays)

#df = df.append(df_jittered)
#df = df.append(df_noised)
#df.to_csv('extended.csv', index=False)

def build_args_parser():
    parser = ArgumentParser(description="Data generator")
    parser.add_argument('--input', help='Path of the file with training data')
    parser.add_argument('--output', help='Path of the output file')
    parser.add_argument('--jitter', help='How many jittered images to generate - multiplier', type=int)    
    parser.add_argument('--sigma', help='Sigma of the jitter', type=float)
    parser.add_argument('--noise', help='How many noised images to generate - multiplier', type=int)
    parser.add_argument('--probability', help='Probability of changing single pixel', type=float)
    return parser

def cli(cli_args):
    parser = build_args_parser()
    args = parser.parse_args(cli_args)
    if args.input and args.output:
        input_df = pd.read_csv(args.input)
        data = transform_data_frame(input_df)
        df_as_arrays = list(map(lambda row: np.array(row), input_df.values))
        df = pd.DataFrame(data = df_as_arrays)
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
        df.to_csv(args.output, index=False)
    else:
        parser.print_usage()
        
cli(sys.argv[1:])




    
