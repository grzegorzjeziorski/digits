# -*- coding: utf-8 -*-
import sys
import pandas as pd
from argparse import ArgumentParser
from sklearn.cross_validation import train_test_split

def build_args_parser():
    parser = ArgumentParser(description="Data generator")
    parser.add_argument('--file', help='Path of the file to split')
    parser.add_argument('--train', help='Path of the file with training data')
    parser.add_argument('--test', help='Path of the file with test data')
    parser.add_argument('--percentage', help='Test data percentage', default=10, type=int)
    parser.add_argument('--dropout', help='Percentage of data to drop', default=0, type=int)    
    return parser

def cli(cli_args):
    parser = build_args_parser()
    args = parser.parse_args(cli_args)
    if args.file and args.train and args.test and args.percentage and args.dropout:
        df = pd.read_csv(args.file)
        new_size = len(df.index) * (100 - args.dropout) / 100        
        df = df[0:new_size]
        train_df, test_df = train_test_split(df, test_size=float(args.percentage)/float(100), random_state=1)
        train_df.to_csv(args.train, index=False)
        test_df.to_csv(args.test, index=False)
    else:
        parser.print_usage()
    
cli(sys.argv[1:])
