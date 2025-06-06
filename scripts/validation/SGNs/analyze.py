import argparse
import pandas as pd

# TODO more logic to separate by annotator etc.
# For now this is just a simple script for global eval

parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

table = pd.read_csv(args.file)
print("Table:")
print(table)

tp = table.tps.sum()
fp = table.fps.sum()
fn = table.fns.sum()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

print("Evaluation:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
