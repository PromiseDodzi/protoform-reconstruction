from ranked_prob_evo.preprocessing import clean_long_vowels
from ranked_prob_evo.evaluator import evaluate_latin_forms, print_summary
import csv
import os

def run():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ipa_file = os.path.join(base_dir, 'romance-ipa.txt')
    multilang_file = os.path.join(base_dir, 'romance-cleaned.txt')
    proto_file = os.path.join(base_dir, 'predicted_protoforms.txt')
    output_file = os.path.join(base_dir, 'merged_output.tsv')
    evaluation_file = os.path.join(base_dir, 'evaluation_base_model.tsv')

    # Clean the input and produce cleaned file in the same directory
    clean_long_vowels(ipa_file)

    with open(multilang_file, 'r', encoding='utf-8') as f:
        multilanguage_lines = [line.strip() for line in f.readlines()]

    with open(proto_file, 'r', encoding='utf-8') as f:
        proto_words = [line.strip() for line in f.readlines()]

    multilang_rows = [line.split('\t') for line in multilanguage_lines]

    header = ["final_proposed_protoform"] + multilang_rows[0]
    merged_rows = [header] + [[pro] + row for pro, row in zip(proto_words, multilang_rows[1:])]

    with open(output_file, 'w', encoding='utf-8') as f:
        for row in merged_rows:
            f.write('\t'.join(row) + '\n')

    results, summary = evaluate_latin_forms(output_file)
    print_summary(summary)

    with open(evaluation_file, "w", encoding="utf-8", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(results)

if __name__ == '__main__':
    run()

