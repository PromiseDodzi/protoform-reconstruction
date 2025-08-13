from ranked_prob_evo.preprocessing import remove_diacritics,pad_words
from ranked_prob_evo.evaluator import levenshtein_distance
from tqdm import tqdm
import csv
import math

def infer_protoforms(words, top_k=10, pad_char='-'): #top_k=10
    """
    Infer multiple protoforms from a list of words.
    Returns a list of (protoform, cost) tuples sorted by cost.
    """
    if not words or all(w == '-' for w in words):
        return [('', float('inf'))] * top_k

    valid_words = [w for w in words if w != '-']
    if not valid_words:
        return [('', float('inf'))] * top_k

    padded_words = pad_words(valid_words, pad_char)
    columns = list(zip(*padded_words))

    protoforms = [([], 0)]

    for col in columns:
        counts = {}
        for c in col:
            if c != pad_char:
                counts[c] = counts.get(c, 0) + 1

        if not counts:
            new_protoforms = []
            for proto, cost in protoforms:
                new_proto = proto + [pad_char]
                new_protoforms.append((new_proto, cost))
            protoforms = new_protoforms
            continue

        candidates = []
        for candidate in counts:
            cost = 0
            for c in col:
                if c == pad_char:
                    cost += 1  
                elif c != candidate:
                    cost += 1  
            candidates.append((candidate, cost))

    
        candidates.sort(key=lambda x: x[1])
        top_candidates = candidates[:top_k]

        # Generate new protoforms by extending with top candidates
        new_protoforms = []
        for proto, proto_cost in protoforms:
            for char, char_cost in top_candidates:
                new_proto = proto + [char]
                new_cost = proto_cost + char_cost
                new_protoforms.append((new_proto, new_cost))

        # Keep only the top k protoforms to prevent combinatorial explosion
        protoforms = sorted(new_protoforms, key=lambda x: x[1])[:top_k]

   
    results = []
    for proto, cost in protoforms:
        proto_str = ''.join(proto).rstrip(pad_char)
        if not proto_str:
            proto_str = ''
        num_missing = len([w for w in words if w == '-'])
        total_cost = cost + num_missing
        results.append((proto_str, total_cost))

    while len(results) < top_k:
        results.append(('', float('inf')))

    return results[:top_k]


def parsimony_proto(input_tsv, output_tsv, top_k=5):
    """Process TSV file to evaluate and select best protoforms"""
    with open(input_tsv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        fieldnames = reader.fieldnames

    new_proto_fieldnames = [f'parsimony_proto{i+1}' for i in range(top_k)]
    output_fields = fieldnames + new_proto_fieldnames

    for row in tqdm(rows, desc="Processing candidate protoforms"):
        processed_row = row.copy()

        for key in processed_row:
            if isinstance(processed_row[key], str):
                processed_row[key] = remove_diacritics(processed_row[key])

        language_cols = [col for col in processed_row.keys()
                         if col not in ['LATIN-CORRECT'] and not col.startswith('Selected_Proto')] 

        attested_words = [str(processed_row[lang]).strip() for lang in language_cols
                         if processed_row[lang] and str(processed_row[lang]).strip().lower() not in ['', 'nan', '-']]



        reconstructed_forms = infer_protoforms(attested_words, top_k)

        for i in range(top_k):
             row[f'parsimony_proto{i+1}'] = reconstructed_forms[i][0] if i < len(reconstructed_forms) else ''


    with open(output_tsv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)



def calculate_pds(total, loss, merger, conditioned, homotopy, complexity):
    """Calculate P(D|S) score"""
    s_loss_log = 0
    for i in range(loss + 1, total + 1):
        s_loss_log += math.log(i, complexity) - math.log(i - loss, complexity)

    s_change_log = merger * math.log(homotopy, complexity)
    s_conditioned_log = conditioned * math.log(2, complexity)

    pds = s_change_log + s_conditioned_log + (s_loss_log * 0.01)- total
    return min(pds / math.log(10, complexity), 0)


def get_possible_changes(proto, daughter):
    """Get all possible changes between proto and daughter forms"""
    changes = []

    # Substitutions
    if len(proto) == len(daughter):
        for i, (p, d) in enumerate(zip(proto, daughter)):
            if p != d:
                changes.extend([
                    (p, d, '', False),  # Unconditional
                    (p, d, '^' if i == 0 else proto[i-1], True),  # Preceding
                    (p, d, '$' if i == len(proto)-1 else proto[i+1], False)  # Following
                ])

    # Deletions
    elif len(proto) - len(daughter) == 1:
        for i, p in enumerate(proto):
            if i >= len(daughter) or p != daughter[i]:
                changes.extend([
                    (p, '', '', False),
                    (p, '', '^' if i == 0 else proto[i-1], True),
                    (p, '', '$' if i == len(proto)-1 else proto[i+1], False)
                ])
                break

    # Insertions
    elif len(daughter) - len(proto) == 1:
        for i, d in enumerate(daughter):
            if i >= len(proto) or d != proto[i]:
                changes.extend([
                    ('', d, '^' if i == 0 else daughter[i-1], True),
                    ('', d, '$' if i == len(daughter)-1 else daughter[i+1], False)
                ])
                break

    return changes


def rank_protoforms(attested, proto_candidates):
    """Rank protoform candidates based on phonological criteria"""
    results = []

    avg_word_length = sum(len(w) for w in attested) / len(attested)
    homotopy = 10
    complexity = 10000
    brevity_weight = 5.0
    edit_distance_weight = 5.0

    for proto in proto_candidates:
        avg_pds = []
        loss = 0
        edit_distances = []

        for daughter in attested:
            all_changes = set()
            if proto == daughter:
                edit_distances.append(0)
                continue

            loss += 1
            for change in get_possible_changes(proto, daughter):
                all_changes.add(change)

            pds = calculate_pds(
                total=len(attested),
                loss=loss,
                merger=len(all_changes),
                conditioned=sum(1 for c in all_changes if c[2] != ''),
                homotopy=homotopy,
                complexity=complexity
            )
            avg_pds.append(pds)
            edit_distances.append(levenshtein_distance(proto, daughter))

        avg_edit_distance = sum(edit_distances) / len(edit_distances)
        edit_penalty = -edit_distance_weight * avg_edit_distance

        length_diff = abs(len(proto) - avg_word_length)
        brevity_penalty = -brevity_weight * length_diff

        base_score = sum(avg_pds) / len(avg_pds) if avg_pds else 0
        adjusted_score = base_score + brevity_penalty + edit_penalty

        results.append({
            'protoform': proto,
            'score': adjusted_score,
            'changes': [f"{c[0]}→{c[1]}/{c[2]}" for c in all_changes],
            'unexplained': loss,
            'avg_edit_distance': avg_edit_distance
        })

    return sorted(results, key=lambda x: x['score'], reverse=True)


def infer_and_rank_protoforms(words, top_k=50): 
    """Generate protoform candidates and rank them"""
    # First generate candidates using parsimony approach
    proto_candidates_with_cost = infer_protoforms(words, top_k=top_k)
    proto_candidates = [proto for proto, cost in proto_candidates_with_cost]

    valid_words = [w for w in words if w and w != '-']
    if not valid_words:
        return []

    # Then rank them using phonological criteria
    ranked_results = rank_protoforms(valid_words, proto_candidates)

    return ranked_results[:top_k]


def evaluate_best_proto(input_tsv, output_tsv, top_ranked):
    """Process TSV file to evaluate and select best protoforms dynamically."""

    with open(input_tsv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        fieldnames = reader.fieldnames


    proto_columns = [f'Selected_Proto{i+1}' for i in range(top_ranked)]

    for row in tqdm(rows, desc="Processing candidate protoforms"):
   
        processed_row = {k: remove_diacritics(v) if isinstance(v, str) else v for k, v in row.items()}

        language_cols = [col for col in processed_row if col != 'LATIN-CORRECT']

   
        attested_words = [
            str(processed_row[lang])
            for lang in language_cols
            if processed_row[lang] and str(processed_row[lang]).lower() != 'nan'
        ]

        # Infer and rank protoforms
        ranked = infer_and_rank_protoforms(attested_words, top_ranked)

     
        for i in range(top_ranked):
            key = f'Selected_Proto{i+1}'
            row[key] = ranked[i]['protoform'] if ranked and i < len(ranked) else ''

    output_fields = fieldnames + proto_columns

    with open(output_tsv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)