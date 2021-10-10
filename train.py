print("import requiring libraries")
import argparse
import sys
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="*", help="file(s), folder, or url to csv")
args = parser.parse_args()

if len(args.path) == 0:
    print("Reading input csv from standard input.")
    input_list = [sys.stdin]
else:
    print("Reading input csv from command line arguments")
    input_list = args.path
