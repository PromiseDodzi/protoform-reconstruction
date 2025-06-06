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


def needleman_wunsch(s1, s2, match_score=1, mismatch_penalty=-1, gap_penalty=-1):
    """
    Needleman-Wunsch algorithm for global sequence alignment.

    Args:
        s1: first string
        s2: second string
        match_score: score for matching characters
        mismatch_penalty: penalty for mismatches
        gap_penalty: penalty for gaps

    Returns:
        tuple of aligned strings
    """
    m, n = len(s1), len(s2)

    score = [[0] * (n + 1) for _ in range(m + 1)]

    traceback = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        score[i][0] = gap_penalty * i
        traceback[i][0] = 1  

    for j in range(1, n + 1):
        score[0][j] = gap_penalty * j
        traceback[0][j] = 2  

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                match = score[i-1][j-1] + match_score
            else:
                match = score[i-1][j-1] + mismatch_penalty

            delete = score[i-1][j] + gap_penalty
            insert = score[i][j-1] + gap_penalty

            max_score = max(match, delete, insert)
            score[i][j] = max_score

            if max_score == match:
                traceback[i][j] = 3 
            elif max_score == delete:
                traceback[i][j] = 1 
            else:
                traceback[i][j] = 2  
 
    aligned1 = []
    aligned2 = []
    i, j = m, n

    while i > 0 or j > 0:
        if traceback[i][j] == 3:  
            aligned1.append(s1[i-1])
            aligned2.append(s2[j-1])
            i -= 1
            j -= 1
        elif traceback[i][j] == 1:  
            aligned1.append(s1[i-1])
            aligned2.append('-')
            i -= 1
        elif traceback[i][j] == 2:  
            aligned1.append('-')
            aligned2.append(s2[j-1])
            j -= 1
        else:  
            break

    return ''.join(reversed(aligned1)), ''.join(reversed(aligned2))