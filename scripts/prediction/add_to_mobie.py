import argparse

from flamingo_tools.mobie import add_raw_to_mobie, add_segmentation_to_mobie
from flamingo_tools.s3_utils import MOBIE_FOLDER


# TODO could also refactor this into flamingo_utils.mobie
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-t", "--type", required=True)
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-k", "--input_key")

    # TODO add optional arguments:
    # - over-ride the mobie folder
    # - ???

    args = parser.parse_args()
    project_folder = MOBIE_FOLDER

    if args.type == "image":
        add_raw_to_mobie(
            mobie_project=project_folder,
            mobie_dataset=args.dataset,
            source_name=args.name,
            input_path=args.input_path,
            input_key=args.input_key,
        )
    elif args.type == "segmentation":
        segmentation_key = "segmentation" if args.input_key is None else args.input_key
        add_segmentation_to_mobie(
            mobie_project=project_folder,
            mobie_dataset=args.dataset,
            source_name=args.name,
            segmentation_path=args.input_path,
            segmentation_key=segmentation_key,
        )
    else:
        raise ValueError(f"Invalid type for a mobie project: {args.name}.")


if __name__ == "__main__":
    main()
