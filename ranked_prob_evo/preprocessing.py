import unicodedata
import os

def remove_diacritics(text):
    "remove tones from segments"

    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def clean_long_vowels(file_path):
    "Removes long vowel markings"
  
    vowels = 'aeiouAEIOU'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        print("The file is empty.")
        return

    headers = lines[0].split()
    num_columns = len(headers)
    data_rows = [line.split() for line in lines[1:]]

    for i, row in enumerate(data_rows):
        if len(row) != num_columns:
            raise ValueError(f"Row {i+2} does not match the number of headers: {row}")

    cleaned_rows = []
    for row in data_rows:
        cleaned_row = []
        for item in row:
            cleaned = ''
            i = 0
            while i < len(item):
                if i + 1 < len(item) and item[i + 1] == 'ː' and item[i] in vowels:
                    cleaned += item[i]  
                    i += 2  
                else:
                    cleaned += item[i]
                    i += 1
            cleaned_row.append(cleaned)
        cleaned_rows.append(cleaned_row)

    folder = os.path.dirname(file_path)
    output_path = os.path.join(folder, 'romance-cleaned.txt')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(headers) + '\n')  
        for row in cleaned_rows:
            f.write('\t'.join(row) + '\n')  

    print(f"Cleaned data written to {output_path}")


def pad_words(words, pad_char='-'):
    """Pad words to equal length with specified character"""
    max_len = max(len(w) for w in words)
    return [w.ljust(max_len, pad_char) for w in words]