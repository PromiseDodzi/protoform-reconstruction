import os
import csv
from tqdm import tqdm
from ranked_prob_evo.preprocessing import clean_long_vowels
from ranked_prob_evo.parsimony import parsimony_proto, evaluate_best_proto
from ranked_prob_evo.rule_transform_and_evolution import DenoisingModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ipa_file = os.path.join(BASE_DIR, 'romance-ipa.txt')
cleaned_file = os.path.join(BASE_DIR, 'romance-cleaned.txt')
parsimony_output = os.path.join(BASE_DIR, 'parsimony_reconstructions.tsv')
selected_proto_file = os.path.join(BASE_DIR, 'selected_proto_results.tsv')
final_output = os.path.join(BASE_DIR, 'final_results.tsv')


def initialize_default_rules(languages):
    """
    Initialize extended sound change rules from Proto language to synchronic languages.
    These are context-aware rules useful for the proto form reconstruction.

    Args:
        languages: List of language names

    Returns:
        Dictionary of rules for each language
    """
    rules = {lang: [] for lang in languages}

    # Romanian rules (Eastern Romance)
    rules['romanian'] = [
        ('ct', 'pt'),     
        ('x', 'ps'),      
        ('gn', 'mn'),     
        ('li', 'i'),      
        ('ce', 'che'),    
        ('ci', 'chi'),    
        ('ge', 'ghe'),
        ('gi', 'ghi'),
        ('qʊ', 'p'),     
        ('v', 'b'),       
        ('cl', 'chi'),    
        ('pl', 'pi'),    
        ('fl', 'fi'),
        ('aʊ', 'o'),      
        #---------------------------------------------14-line rule
        # ('e', 'ie'),      
        # ('ʊm', ''),       
    ]

    # French rules (Gallo-Romance)
    rules['french'] = [
        ('ca', 'cha'),    
        ('co', 'cho'),    
        ('ga', 'ja'),     
        ('ct', 'it'),     
        ('pt', 'it'),     
        ('gn', 'gn'),     
        ('li', 'j'),      
        ('ll', 'l'),      
        ('pl', 'bl'),     
        ('fl', 'bl'),     
        ('aʊ', 'o'),     
        ('o', 'eu'),      
        ('e', 'ie'),     
        ('a', 'e'),      
        #--------------------------------------------14-rule line
        # ('s', ''),        
        # ('ʊm', ''),      
    ]

    # Italian rules (Italo-Romance)
    rules['italian'] = [
        ('cl', 'chi'),   
        ('ct', 'tt'),     
        ('pt', 'tt'),     
        ('gn', 'gn'),     
        ('x', 'ss'),      
        ('qʊ', 'qu'),     
        ('li', 'gl'),     
        ('pl', 'pi'),    
        ('fl', 'fi'),     
        ('aʊ', 'o'),     
        ('o', 'o'),       
        ('e', 'e'),       
        ('a', 'a'),
        ('ʊm', 'o'),      
       # -------------------------------------------- 14-rule line
        # ('ʊs', 'o'),      
        # ('ʊ', 'o')
    ]

    # Spanish rules (Western Ibero-Romance)
    rules['spanish'] = [
        ('ct', 'ch'),    
        ('li', 'j'),      
        ('cl', 'll'),     
        ('pl', 'll'),     
        ('fl', 'll'),     
        ('gn', 'ñ'),      
        ('f', 'h'),       
        ('aʊ', 'o'),      
        ('o', 'ue'),      
        ('e', 'ie'),     
        ('qʊ', 'c'),      
        ('v', 'b'),       
        ('s', 's'),
        ('ʊm', 'o'),
        #--------------------------------- 14-rule line
        # ('ʊs', 'o'),
        # ('ʊ', 'o')
    ]

    # Portuguese rules (Galician-Portuguese / Lusitanian)
    rules['portuguese'] = [
        ('ct', 'it'),     
        ('pl', 'ch'),     
        ('fl', 'ch'),     
        ('cl', 'ch'),     
        ('li', 'lh'),     
        ('ni', 'nh'),    
        ('gn', 'nh'),     
        ('ce', 'ce'),     
        ('ci', 'ci'),
        ('qʊ', 'c'),      
        ('l', 'r'),       
        ('aʊ', 'ou'),    
        ('o', 'o'),
        ('e', 'ei'),     
        #----------------------------------------------- 14-rule line
        # ('a', 'a'),
        # ('ʊm', 'o')
    ]

    return rules


def read_tsv(filename):
    """Read TSV file and return list of dictionaries"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

def write_tsv(data, filename, expected_columns):
    """Write list of dictionaries to TSV file"""
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=expected_columns, delimiter='\t')
        writer.writeheader()
        writer.writerows(data)


def process_data(data, languages, rules, phylogeny):
    """
    Process data rows with seed protoforms to find optimal Protoform reconstructions.

    Args:
        data: List of dictionaries containing the cognate data
        rules: Dictionary of sound change rules for each language
        phylogeny: Newick format string representing language relationships

    Returns:
        List of processed data rows with 'final_proposed_protoform' and scores
    """

    model = DenoisingModel(rules=rules, languages=languages, phylogeny=phylogeny)

    for row in tqdm(data, desc="Predicting protoforms", unit="row"):
        final_proto = model.analyze_row(row)
        row['final_proposed_protoform'] = final_proto if final_proto else '-'

        # Calculate score for the proposed protoform
        if final_proto:
            row['score'] = model.score_protoform(final_proto, {
                lang: row.get(lang.upper(), '-')
                for lang in languages
            })
        else:
            row['score'] = float('-inf')

    return data


def main(input_file, output_file):
    """Main function to process the data"""

    print(f"Reading data from {input_file}")
    data = read_tsv(input_file)

    # Initialize the model with appropriate sound change rules
    languages = ['romanian', 'french', 'italian', 'spanish', 'portuguese']
    rules = initialize_default_rules(languages)
    phylogeny = "((French,Spanish),Portuguese,Italian,Romanian);"

    print(f"Processing {len(data)} rows of data")
    processed_data = process_data(data=data, languages=languages, rules=rules, phylogeny=phylogeny)

    # Dynamically collect all columns from processed data
    expected_columns = set()
    for row in processed_data:
        expected_columns.update(row.keys())
    expected_columns = list(expected_columns)

    print(f"Writing results to {output_file}")
    write_tsv(processed_data, output_file, expected_columns)
    print("Processing complete!")

    return processed_data


if __name__=='__main__':
    clean_long_vowels(ipa_file)
    parsimony_proto(cleaned_file, parsimony_output)
    evaluate_best_proto(cleaned_file, selected_proto_file, top_ranked=5)
    results = main(selected_proto_file, final_output)