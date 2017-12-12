import argparse

def merge(input_files, output_file):
    texts = set()
    for in_f in input_files:
        with open(in_f, 'r') as f:
            for line in f:
                texts.add(line)

    with open(output_file, 'w') as f:
        f.writelines(texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()
    input_files = args.input_files.split()
    print(input_files)
    merge(input_files, args.output_file)
    