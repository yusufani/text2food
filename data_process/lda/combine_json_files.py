import argparse
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', help='Output JSONL file path')
    parser.add_argument('--num_files', type=int, default=None, help='Number of JSONL files to process')
    return parser.parse_args()


def combine_jsonl_files(output_file, num_files=None):
    input_dir = os.getcwd()  # Set input_dir to the current directory
    jsonl_files = glob.glob(os.path.join(input_dir, '*.jsonl'))

    if num_files is not None and num_files < len(jsonl_files):
        jsonl_files = jsonl_files[:num_files]

    with open(output_file, 'w') as outfile:
        for file_path in jsonl_files:
            with open(file_path) as infile:
                for line in infile:
                    outfile.write(line)


def main():
    args = parse_args()
    combine_jsonl_files(args.output_file, args.num_files)
    if args.num_files is not None:
        print(f"Combined {args.num_files} JSONL files into {args.output_file}")
    else:
        print(f"Combined {len(glob.glob(os.path.join(os.getcwd(), '*.jsonl')))} JSONL files into {args.output_file}")


if __name__ == '__main__':
    main()
