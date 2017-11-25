import argparse
from buildpretrainemb import save_dict

def load_element_keywords(path):
    element_keywords = []
    with open(path, 'r') as f:
        for line in f:
            keywords = line.split('\t')[1].split()
            element_keywords.append(keywords)
    return element_keywords


def get_element_vector(data_path, element_keywords, output):
    element_vector = dict()
    offset = len(element_keywords)
    with open(data_path, 'r') as f:
        for line in f:
            line_list = line.split('\t')
            id = "train_"+line_list[0]
            text = ''.join(line_list[1].split())
            element_vector[id] = []
            for ei, element in enumerate(element_keywords):
                has_cur_e = False
                for e in element:
                    if e in text:
                        # has this element, index=ei
                        element_vector[id].append(ei)  
                        has_cur_e = True
                        break
                if not has_cur_e:
                    # doesn't has, index=ei+len(element_keywords)
                    element_vector[id].append(ei + offset)  

    print(element_vector['train_60'], len(element_vector['train_60']))
    print(element_vector['train_355'], len(element_vector['train_355']))

    save_dict(element_vector, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--element-keywords-path", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    element_keywords = load_element_keywords(args.element_keywords_path)
    get_element_vector(args.data_path, element_keywords, args.output)
