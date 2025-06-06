import argparse
import pandas as pd


def compute_scores(table, annotator=None):
    if annotator is None:
        annotator = "all"
    else:
        table = table[table.annotator == annotator]

    tp = table.tps.sum()
    fp = table.fps.sum()
    fn = table.fns.sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    return pd.DataFrame({
        "annotator": [annotator], "precision": [precision], "recall": [recall], "f1-score": [f1_score]
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file")
    args = parser.parse_args()

    table = pd.read_csv(args.result_file)
    annotators = pd.unique(table.annotator)

    results = []
    for annotator in annotators:
        scores_annotator = compute_scores(table, annotator)
        results.append(scores_annotator)
    results.append(compute_scores(table, annotator=None))

    results = pd.concat(results)
    print(results)


if __name__ == "__main__":
    main()
