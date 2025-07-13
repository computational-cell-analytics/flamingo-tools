import argparse
import sys

import torch
from torch_em.util import load_model

sys.path.append("/home/pape/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/pape41/u12086/Work/my_projects/czii-protein-challenge")


def export_model(input_, output):
    model = load_model(input_, device="cpu")
    torch.save(model, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    export_model(args.input, args.output)


if __name__ == "__main__":
    main()
