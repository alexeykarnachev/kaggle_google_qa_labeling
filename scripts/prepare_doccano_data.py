import json
import argparse
from html import escape
from pathlib import Path
from xml.etree import cElementTree as ElementTree

from bs4 import BeautifulSoup as bs


class XmlListConfig(list):
    def __init__(self, xml_list):
        super().__init__()
        for element in xml_list:
            if element:
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                txt = element.text.strip()
                if txt:
                    self.append(txt)


class XmlDictConfig(dict):
    def __init__(self, parent_element):
        super().__init__()
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                if len(element) == 1 or element[0].tag != element[1].tag:
                    dictionary = XmlDictConfig(element)
                else:
                    dictionary = {element[0].tag: XmlListConfig(element)}
                if element.items():
                    dictionary.update(dict(element.items()))
                self.update({element.tag: dictionary})

            elif element.items():
                self.update({element.tag: dict(element.items())})
            else:
                self.update({element.tag: element.text})


parser = argparse.ArgumentParser(description='Prepare jsonl files for Doccano labeling')
parser.add_argument('--stackexchange_dir', type=Path, help='Directory with stackexchange files (with Posts.xml inside)')
parser.add_argument('--out_dir', type=Path, help='Output directory (.jsonl files will be saved there)')
parser.add_argument('--docs_per_file', type=int, default=1000, help='Number of documents per file')

args = parser.parse_args()

out_dir = args.out_dir
stackexchange_dir = args.stackexchange_dir
docs_per_file = args.docs_per_file


if __name__ == '__main__':
    out_dir.mkdir(parents=True, exist_ok=True)
    inp_file = stackexchange_dir / 'Posts.xml'

    tree = ElementTree.parse(inp_file)
    f = (out_dir / '0.jsonl').open('w')

    i = 0
    for x in tree.iter():
        d = XmlDictConfig(x)
        row = d.get('row', d)
        text = bs(row['Body']).text

        if len(text) > 5:
            text = escape(text, quote=False)
            f.write(json.dumps({'text': text}) + '\n')
            i += 1

            if i % docs_per_file == 0:
                f.close()
                f = (out_dir / f'{i}.jsonl').open('w')
