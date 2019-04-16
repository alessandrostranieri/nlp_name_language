import glob
import os
import string
import unicodedata
from random import shuffle, seed
from typing import List, Tuple, Set

DATA_FILE_PATH = 'data/names/*.txt'

all_letters = string.ascii_letters + " .,;'"


def find_files(path):
    return glob.glob(path)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_lines(file_name: str) -> List[str]:
    file_lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(file_line) for file_line in file_lines]


name_language_list: List[Tuple[str, str]] = []

for filename in find_files(DATA_FILE_PATH):
    category = os.path.splitext(os.path.basename(filename))[0]
    lines = read_lines(filename)

    for line in lines:
        name_language_list.append((line, category))

languages = set()


def write_to_file(name_language: List[Tuple[str, str]], file_name: str) -> None:
    with open(file_name, mode='w') as f:
        for name, language in name_language:
            f.write(f'{name} {language}\n')
            languages.add(language)


num_samples = len(name_language_list)
split_point = int(float(num_samples) * 0.7)

seed(42)
shuffle(name_language_list)
# Write files
write_to_file(name_language_list[:split_point], 'data/training.txt')
write_to_file(name_language_list[split_point:], 'data/validation.txt')

print(languages)