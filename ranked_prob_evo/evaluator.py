from ranked_prob_evo.preprocessing import remove_diacritics
import csv

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insert = previous_row[j + 1] + 1
            delete = current_row[j] + 1
            substitute = previous_row[j] + (c1 != c2)
            current_row.append(min(insert, delete, substitute))
        previous_row = current_row

    return previous_row[-1]


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


def count_correct_characters(s1, s2):
    """
    Obtain a count of the exact number of matching character between two aligned strings.

    Args:
        s1: string 1
        s2: string 2

    Returns:
        A scalar indicating the count of matching characters.
    """
    aligned_s1, aligned_s2= needleman_wunsch(s1, s2)
    correct_count=0
    for p, t in zip(aligned_s1, aligned_s2):
        if p==t and p != '-':
            correct_count += 1
    return correct_count


def compute_cer(s1, s2):
    """
    obtain character error rate between two aligned words

    Args:
        s1: string 1
        s2: string 2

    Returns:
        A scalar indicating the caracter error rate
    """
    aligned_s1, aligned_s2 = needleman_wunsch(s1, s2)
    subs=0
    ins=0
    dels=0

    for p, t in zip(aligned_s1, aligned_s2):
        if p == t:
            continue
        elif p== '-':
            dels +=1
        elif t == '-':
            ins += 1
        else:
            subs +=1

    ref_length = len(s2)
    cer = (subs + ins + dels) / ref_length  if ref_length > 0 else 0
    return cer

def get_phonological_features(char):
    """
    Get phonological features for a character using articulatory features.

    Args:
        char: a character

    Returns:
        a dictionary that contains articulatory features
    """

    features = {}
    # Vowel features
    if char.lower() in 'aeiouəɨɯɪʊɛɔæɑɒʌɤøyɵœɶɐɘɜɞʉɯɪʏʊɨʉɘɜɤɚɾʐɭɫʎʝ':
        features['type'] = 'vowel'
        # Height
        if char.lower() in 'iɨɯɪʏʊeøyɵəɘɵ':
            features['height'] = 'high'
        elif char.lower() in 'eɛøœəɘɜɞ':
            features['height'] = 'mid'
        elif char.lower() in 'aæɑɒɐɶ':
            features['height'] = 'low'
        else:
            features['height'] = 'other'

        # Backness
        if char.lower() in 'iɪeɛæyʏøœɶ':
            features['backness'] = 'front'
        elif char.lower() in 'ɨəɘɜɞɵ':
            features['backness'] = 'central'
        elif char.lower() in 'ɯʊoɔɑɒʌɤɐ':
            features['backness'] = 'back'
        else:
            features['backness'] = 'other'

        # Roundedness
        if char.lower() in 'ouɔɒøyʏɵœʊ':
            features['rounded'] = 'rounded'
        else:
            features['rounded'] = 'unrounded'

        # Nasalization
        if char.lower() in 'ãẽĩõũ':
            features['nasalized'] = True
        else:
            features['nasalized'] = False

    # Consonant features
    else:
        features['type'] = 'consonant'

        # Voicing
        if char.lower() in 'bdgjlmnrvwzðʒɮʐɾɦɢʁɰʕʔɣɹɻʀʍ':
            features['voicing'] = 'voiced'
        else:
            features['voicing'] = 'voiceless'

        # Manner of articulation
        if char.lower() in 'bdptdkgqʔ':
            features['manner'] = 'stop'
        elif char.lower() in 'fvθðszʃʒxɣχʁħʕh':
            features['manner'] = 'fricative'
        elif char.lower() in 'mnŋɲɳɴ':
            features['manner'] = 'nasal'
        elif char.lower() in 'ʦʣʧʤ':
            features['manner'] = 'affricate'
        elif char.lower() in 'rlɹɾɽʀ':
            features['manner'] = 'liquid'
        elif char.lower() in 'jw':
            features['manner'] = 'glide'
        else:
            features['manner'] = 'other'

        # Place of articulation
        if char.lower() in 'bpmfvw':
            features['place'] = 'labial'
        elif char.lower() in 'tdnszlrθð':
            features['place'] = 'dental/alveolar'
        elif char.lower() in 'ʃʒʧʤ':
            features['place'] = 'postalveolar'
        elif char.lower() in 'jɲçʝ':
            features['place'] = 'palatal'
        elif char.lower() in 'kgŋxɣ':
            features['place'] = 'velar'
        elif char.lower() in 'qχʁ':
            features['place'] = 'uvular'
        elif char.lower() in 'ħʕʜʢ':
            features['place'] = 'pharyngeal'
        elif char.lower() in 'ʔh':
            features['place'] = 'glottal'
        else:
            features['place'] = 'other'

    return features

def feature_diff(char1, char2):
    """
    Calculate the feature difference between two characters.

    Args:
        char1: first character
        char2: second character
    Returns:
        a value between 0 (identical) and 1 (completely different).
    """

    if char1 == char2:
        return 0.0

    # Get features for both characters
    f1 = get_phonological_features(char1)
    f2 = get_phonological_features(char2)

    if f1.get('type') != f2.get('type'):
        return 1.0

    common_keys = set(f1.keys()) & set(f2.keys())
    if not common_keys:
        return 1.0

    matches = sum(1 for k in common_keys if f1[k] == f2[k])
    feature_similarity = matches / len(common_keys)

    return 1.0 - feature_similarity


def feature_edit_distance(word1, word2):
    """
    Calculate the feature-weighted edit distance between two words.

    Args:
        word1: first word
        word2: second word

    Returns:
        A scalar that indicates the feature weighted edit distance between the two words
    """

    if not word1 and not word2:
        return 0.0

    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            # Substitution cost based on feature difference
            if word1[i-1] == word2[j-1]:
                subst_cost = 0
            else:
                subst_cost = feature_diff(word1[i-1], word2[j-1])

            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + subst_cost
            )

    return dp[m][n]

def is_vowel(char):
    """Check if a character is a vowel (with or without diacritics)."""
    base_char = char.lower()
    vowels = {
        'a', 'e', 'i', 'o', 'u', 'y','ʊ','ɐ', 'ɨ','ə','ɛ','ɔ','ɪ',  
        'ā', 'ē', 'ī', 'ō', 'ū', 'ȳ',  
        'ă', 'ĕ', 'ĭ', 'ŏ', 'ŭ',       
        'á', 'é', 'í', 'ó', 'ú', 'ý',  
        'à', 'è', 'ì', 'ò', 'ù',       
        'â', 'ê', 'î', 'ô', 'û',      
        'ä', 'ë', 'ï', 'ö', 'ü', 'ÿ'  
    }
    return base_char in vowels

def is_phonotactically_valid(word):
    """Check for invalid Latin consonant clusters."""
    forbidden_clusters = {
        # Illegal onsets
        'tk', 'pk', 'gb', 'bm', 'bn', 'dn', 'ln', 'nr', 'nm', 'mr',
        'pf', 'bv', 'ǧh', 'ḱf', 'ḷp', 'gd', 'kp', 'pt', 'fp',

        # Illegal codas
        'mt', 'md', 'mk', 'mg', 'mh', 'nb', 'nd', 'ng', 'nh',
        'rl', 'rm', 'rn', 'rs', 'rt', 'rv', 'rz',

        # Cross-syllable violations
        'mm', 'nn', 'bb', 'dd', 'gg', 'pp', 'tt', 'kk',
        'vv', 'ff', 'ss', 'zz', 'xx', 'hh',

        # Rare/avoided sequences
        'tl', 'dl', 'ts', 'dz', 'ks', 'ps', 'bs', 'gs',
        'ḑh', 'ẑh', 'ṡh', 'ṫh', 'ḣh',

        # Greek borrowings 
        'rh', 'th', 'ph', 'ch', 'kh', 'yh'
    }
    word_lower = word.lower()
    return not any(
        word_lower[i:i+2] in forbidden_clusters
        for i in range(len(word_lower)-1)
    )

def evaluate_latin_forms(tsv_file):
    """
    Evaluate predicted Latin forms against correct forms with:
    - Consonant/vowel error tracking
    - Phonotactic validity checks
    - All original metrics
    """
    # Initialize counters
    results = []
    total_correct_chars = 0
    total_chars = 0
    total_feature_distance = 0
    total_feature_elements = 0
    exact_matches = 0
    total_consonant_errors = 0
    total_vowel_errors = 0
    phonotactic_violations = 0

    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            true_word = row['LATIN-CORRECT'].strip()
            pred_word = row['final_proposed_protoform'].strip()  #
            true_word_clean = remove_diacritics(true_word)
            pred_word_clean = remove_diacritics(pred_word)

            if not true_word_clean or not pred_word_clean:
                continue

            # Alignment and basic metrics
            aligned_pred, aligned_true = needleman_wunsch(pred_word_clean, true_word_clean)
            edit_dist = levenshtein_distance(pred_word_clean, true_word_clean)
            cer = compute_cer(pred_word_clean, true_word_clean)

            # Character-level analysis
            correct_chars = 0
            consonant_errors = 0
            vowel_errors = 0

            for t, p in zip(aligned_true, aligned_pred):
                if t == p and p != '-':
                    correct_chars += 1
                elif p != t:
                    if t == '-':
                        continue  # insertion, counted in CER
                    elif p == '-':
                        if is_vowel(t):
                            vowel_errors += 1
                        else:
                            consonant_errors += 1
                    else:
                        if is_vowel(t):
                            vowel_errors += 1
                        else:
                            consonant_errors += 1

            # Update counters
            total_correct_chars += correct_chars
            total_chars += len(true_word_clean)
            total_consonant_errors += consonant_errors
            total_vowel_errors += vowel_errors

            # Phonotactic check
            phonotactic_valid = is_phonotactically_valid(pred_word_clean)
            if not phonotactic_valid:
                phonotactic_violations += 1

            # Feature metrics
            feature_dist = feature_edit_distance(pred_word_clean, true_word_clean)
            total_feature_distance += feature_dist
            total_feature_elements += len(true_word_clean) * 5  # 5 features/char

            results.append({
                'true_form': true_word,
                'predicted_form': pred_word,
                'edit_distance': edit_dist,
                'normalized_edit_distance': edit_dist / max(len(pred_word_clean), len(true_word_clean)),
                'correct_chars': correct_chars,
                'consonant_errors': consonant_errors,
                'vowel_errors': vowel_errors,
                'phonotactic_valid': phonotactic_valid,
                'char_error_rate': cer,
                'feature_distance': feature_dist,
                'normalized_feature_distance': feature_dist / (len(true_word_clean) * 5),
                'exact_match': int(pred_word_clean == true_word_clean)
            })

    # Calculate summary statistics
    num_examples = len(results)
    if num_examples == 0:
        return [], {}

    summary = {
        'num_examples': num_examples,
        'char_accuracy': round(total_correct_chars / total_chars, 4) if total_chars > 0 else 0,
        'word_accuracy': round(sum(r['exact_match'] for r in results) / num_examples, 4),
        'consonant_error_rate': round(total_consonant_errors / total_chars, 4) if total_chars > 0 else 0,
        'vowel_error_rate': round(total_vowel_errors / total_chars, 4) if total_chars > 0 else 0,
        'phonotactic_violation_rate': round(phonotactic_violations / num_examples, 4),
        'mean_edit_distance': round(sum(r['edit_distance'] for r in results) / num_examples, 4),
        'mean_normalized_edit_distance': round(sum(r['normalized_edit_distance'] for r in results) / num_examples, 4),
        'mean_feature_distance': round(total_feature_distance / num_examples, 4),
        'feature_error_rate': round(total_feature_distance / total_feature_elements, 4) if total_feature_elements > 0 else 0,
        'mean_char_error_rate': round(sum(r['char_error_rate'] for r in results) / num_examples, 4)
    }

    return results, summary

def print_summary(summary):
    """Print the evaluation summary in a readable format."""
    print("\nEvaluation Summary:")
    print("------------------")
    print(f"Number of examples: {summary['num_examples']}")
    print(f"Character accuracy: {summary['char_accuracy']:.2%}")
    print(f"Word accuracy: {summary['word_accuracy']:.2%}")
    print(f"Consonant error rate: {summary['consonant_error_rate']:.2%}")
    print(f"Vowel error rate: {summary['vowel_error_rate']:.2%}")
    print(f"Phonotactic violation rate: {summary['phonotactic_violation_rate']:.2f}")
    print(f"Mean edit distance: {summary['mean_edit_distance']:.2f}")
    print(f"Mean normalized edit distance: {summary['mean_normalized_edit_distance']:.2f}")
    print(f"Mean feature distance: {summary['mean_feature_distance']:.2f}")
    print(f"Feature error rate: {summary['feature_error_rate']:.2f}")
    print(f"Mean character error rate: {summary['mean_char_error_rate']:.2f}")

if __name__ == "__main__":
    tsv_file = 'ranked_prob_evo/final_results.tsv' 
    results, summary = evaluate_latin_forms(tsv_file)
    print_summary(summary)
   
    with open("ranked_prob_evo/evaluation_results_adopted.tsv", "w", encoding="utf-8", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(results)




