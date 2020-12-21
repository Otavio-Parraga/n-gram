import re


def clean_sentence(sentence):
    sentence = split_punctuations(sentence)
    sentence = remove_selected_chars(sentence, '\n')
    return str(sentence).lower()


def split_punctuations(sentence):
    sentence = re.findall(r"[\w'-]+|[.,!?;]", sentence)
    return ' '.join(sentence)


def remove_selected_chars(sentence, chars):
    return sentence.replace(chars, sentence)
