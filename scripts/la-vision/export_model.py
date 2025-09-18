import argparse
import sys

import torch
from torch_em.util import load_model

sys.path.append("/home/pape/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/pape41/u12086/Work/my_projects/czii-protein-challenge")
sys.path.append("../synapse_marker_detection")


def export_model(input_, output, latest):
    model = load_model(input_, device="cpu", name="latest" if latest else "best")
    torch.save(model, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    export_model(args.input, args.output, latest=args.latest)


if __name__ == "__main__":
    main()
