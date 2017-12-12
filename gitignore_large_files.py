import os
import argparse

def main(args):
    filepath = os.popen('find . -size +{}'.format(args.threshold)).read().split('\n')
    gitignore_filepath = []
    for fp in filepath:
        if fp.startswith('./.git/'):
            continue
        else:
            gitignore_filepath.append(fp[2:])  # Remove './' before each path.
    os.system('rm -f .gitignore')
    os.system('cp .gitignore.bak .gitignore')

    with open('.gitignore', 'a') as f:
        f.write('\n# Files larger than {}\n'.format(args.threshold))
        f.write('\n'.join(gitignore_filepath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=str, default='500k')
    args = parser.parse_args()

    main(args)
