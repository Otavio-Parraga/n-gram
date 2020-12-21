import argparse
from NGramModel import NGramModel

parser = argparse.ArgumentParser()

parser.add_argument('--size', type=int, default=100)
parser.add_argument('--file_name', type=str)
parser.add_argument('--text', type=str, default='o')
parser.add_argument('--limit', type=int, default=10)

args = parser.parse_args()

model = NGramModel(f'./data/{args.file_name}.txt')


def generate_text(begin, size, limit):
    text = f'{begin} '
    for i in range(size):
        text += model.find_next_word(text, limit)[0] + ' '
    return text


if __name__ == '__main__':
    generated_text = generate_text(args.text, args.size, args.limit)
    print(generated_text)