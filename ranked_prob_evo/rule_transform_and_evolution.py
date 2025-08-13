from ranked_prob_evo.evaluator import levenshtein_distance
from collections import Counter
import itertools
import math


class DenoisingModel:
    def __init__(self, rules=None, languages=None, phylogeny=None):
        self.rules = rules or {}  
        self.languages = languages or []  
        self.protoform_counter = Counter()
        self.phylogeny = self._parse_newick(phylogeny) if phylogeny else None

        # Initialize phylogenetic weights if tree is provided
        self.phylogenetic_weights = self._calculate_phylogenetic_weights() if self.phylogeny else None

        #phonetic features
        self.vowels= {'a', 'e', 'i', 'o', 'u', 'ʊ', 'ɛ', 'ɨ'}
        self.consonants= set('bβcdfghjklmnpqrstvxzʤɹ')
        self.stops = {'p', 'b', 't', 'd', 'c', 'g', 'k', 'q'}
        self.fricatives = {'f', 'v', 's', 'z', 'h'}
        self.liquids = {'l', 'r'}
        self.nasals = {'m', 'n'}
        # Phoneme frequency statistics can be computed as priors. 
        # In our model, we do not use this as we observe degraded performance
        self.latin_phoneme_freq = {}

        # Morphological cues and phonotactic constraints
        self.common_endings = ['ʊm', 'ʊs', 'a','ɪs', 'is','ɨs', 'em', 'am', 'or', 'er', 'es', 'ae', 'iʊm', 'iʊm', 'ɨʊm', 'ɨo', 'io', 'ia','ɨa', 'id', 'ɨd', 'ɛm']
        self.common_prefixes = ['ad', 'con', 'de', 'dis', 'ex', 'in', 'per', 'pro', 're', 'sub', 'trans', 'ab', 'ob']
        self.vowel_patterns = ['a-a', 'e-e', 'i-i', 'ɨ-ɨ', 'o-o', 'e-a', 'i-a', 'i-e', 'u-a', 'ʊ-ʊ']
        self.valid_consonant_clusters = [
            'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'qu', 'sc', 'sp', 'st', 'tr',
            'ct', 'nt', 'rt', 'lt', 'nd', 'mb', 'mp', 'ns', 'nf', 'rx', 'ps', 'pt', 'bs', 'mn', 'gn',
        ]
        self.invalid_sequences = ['ll','bm', 'fn', 'gm', 'kn', 'pf', 'sr', 'tl', 'wl', 'zb','ws', 'gk', 'tʃ']

        # Create reverse rules with weights 
        self.reverse_rules = self._initialize_reverse_rules()

    def _initialize_reverse_rules(self):
        """Initialize reverse rules with weights, handling empty rules case"""
        reverse_rules = {}
        if self.rules:
            for lang, rule_list in self.rules.items():
                reverse_rules[lang] = [(tgt, src, self._rule_weight(src, tgt)) for src, tgt in rule_list]
        return reverse_rules

    def _parse_newick(self, newick_str):
        """Parse Newick format string into a tree structure"""
        if not newick_str:
            return None
        
        stack = []
        current_node = {}
        i = 0
        n = len(newick_str)

        while i < n:
            char = newick_str[i]
            if char == '(':
                new_node = {'children': [], 'parent': current_node}
                if current_node:
                    current_node['children'].append(new_node)
                stack.append(current_node)
                current_node = new_node
                i += 1
            elif char == ')':
                if stack:
                    current_node = stack.pop()
                i += 1
            elif char == ',':
                i += 1
            elif char == ';':
                break
            else:
                label = []
                while i < n and newick_str[i] not in '(),;':
                    label.append(newick_str[i])
                    i += 1
                current_node['name'] = ''.join(label).strip()

        return current_node

    def _calculate_phylogenetic_weights(self):
        """Calculate weights based on phylogenetic position"""
        if not self.phylogeny:
            return None

        weights = {}

        def traverse(node, depth=0):
            if 'name' in node and node['name'] in self.languages:
                weights[node['name']] = 1.0 + (1.0 / (depth + 1))
            for child in node.get('children', []):
                traverse(child, depth + 1)

        traverse(self.phylogeny)

        total = sum(weights.values())
        if total > 0:
            for lang in weights:
                weights[lang] /= total

        return weights

    def _rule_weight(self, src, tgt):
        """Calculate weight for a rule based on phonological distance"""
        if not src or not tgt:
            return 0.01

        weight = 1.0 / (1 + abs(len(src) - len(tgt)) * 0.5)

        if len(src) == 1 and len(tgt) == 1:
            if (src in self.vowels and tgt in self.vowels) or \
                    (src in self.stops and tgt in self.stops) or \
                    (src in self.fricatives and tgt in self.fricatives) or \
                    (src in self.liquids and tgt in self.liquids) or \
                    (src in self.nasals and tgt in self.nasals):
                weight *= 1.5

        return weight

    def train(self, protoforms):
        """Build prior protoform frequencies""" #NB: doing this degrades our model's performance
        self.protoform_counter.update(protoforms)
        for proto in protoforms:
            for c in proto:
                if c in self.latin_phoneme_freq:
                    self.latin_phoneme_freq[c] += 0.01

    def apply_rules(self, protoform, lang):
        """Apply derivation rules to get modern word"""
        if not self.rules or lang not in self.rules:
            return protoform  

        derived = protoform
        for src, tgt in self.rules.get(lang, []):
            derived = derived.replace(src, tgt)
        return derived

    def reverse_apply_rules(self, word, lang, observed_forms=None, max_depth=5):
        """Reverse-apply rules recursively with length awareness removed"""
        if not word or word == '-':
            return set()
        if not self.reverse_rules or lang not in self.reverse_rules:
            return {word}

        all_candidates = {word}
        current_candidates = {word}
        MAX_CANDIDATES = 2000

        for depth in range(max_depth):
            new_candidates = set()

            for candidate in current_candidates:
                for tgt, src, _ in sorted(self.reverse_rules.get(lang, []),
                                            key=lambda x: -x[2])[:15]:
                    if tgt not in candidate:
                        continue

                    pos = -1
                    while True:
                        pos = candidate.find(tgt, pos + 1)
                        if pos == -1:
                            break

                        new_candidate = candidate[:pos] + src + candidate[pos + len(tgt):]

                        if not any(invalid in new_candidate
                                    for invalid in self.invalid_sequences):
                            new_candidates.add(new_candidate)

                        if pos + len(tgt) == len(candidate):
                            for ending in self.common_endings:
                                extended = new_candidate + ending
                                if not any(invalid in extended
                                            for invalid in self.invalid_sequences):
                                    new_candidates.add(extended)

                    if len(new_candidates) > MAX_CANDIDATES:
                        break

            all_candidates.update(new_candidates)
            if not new_candidates or len(all_candidates) > MAX_CANDIDATES:
                break
            current_candidates = new_candidates

        return self._filter_by_form_likeness(all_candidates, MAX_CANDIDATES)

    def beam_search(self, initial_protoforms, observed_forms, beam_width=3, max_iter=5):
        """Perform beam search starting from initial protoforms"""
        beam = [(proto, self.score_protoform(proto, observed_forms))
                for proto in initial_protoforms]
        beam.sort(key=lambda x: -x[1])
        beam = beam[:beam_width]

        for _ in range(max_iter):
            new_candidates = []

            for proto, _ in beam:
                variants = self.generate_variants(proto)
                for variant in variants:
                    score = self.score_protoform(variant, observed_forms)
                    new_candidates.append((variant, score))

            combined = beam + new_candidates
            combined.sort(key=lambda x: -x[1])
            beam = combined[:beam_width]

            # Early stopping if no improvement
            if beam[0][1] <= combined[0][1]:
                break

        return beam

    def generate_variants(self, protoform):
        """Generate plausible variants of a protoform"""
        variants = set()

        # Vowel alternations
        for i, c in enumerate(protoform):
            if c in self.vowels:
                for v in self.vowels:
                    if v != c:
                        variant = protoform[:i] + v + protoform[i+1:]
                        variants.add(variant)

        #Common suffix alternations
        for ending in self.common_endings:
            if protoform.endswith(ending):
                for alt_ending in self.common_endings:
                    if alt_ending != ending:
                        variant = protoform[:-len(ending)] + alt_ending
                        variants.add(variant)

        #Consonant cluster simplifications
        for cluster in self.valid_consonant_clusters:
            if cluster in protoform:
                pos = protoform.find(cluster)
                variant = protoform[:pos] + cluster[-1] + protoform[pos+len(cluster):]
                variants.add(variant)

        #Length variations (±1 character)
        if len(protoform) > 3:
            for i in range(len(protoform)):
                variant = protoform[:i] + protoform[i+1:]
                variants.add(variant)

            for i in range(len(protoform)):
                variant = protoform[:i] + protoform[i] + protoform[i:]
                variants.add(variant)

        return variants

    def score_protoform(self, protoform, observed_forms):
        """Score a protoform using P(cognates|proto) * P(proto)"""
        if not protoform:
            return float('-inf')

        likelihood = self.likelihood(protoform, observed_forms)
        
        prior = self.prior(protoform)

        return likelihood + prior

    def get_consensus_form(self, observed_forms):
        """Generate consensus protoform with phylogenetic awareness"""
        if not observed_forms:
            return None

        # Generate candidates per language with phylogenetic weighting
        lang_candidates = {}
        for lang, word in observed_forms.items():
            if word and word != '-':
                candidates = self.reverse_apply_rules(word, lang, observed_forms)
                if candidates:
                    weight = self.phylogenetic_weights.get(lang, 1.0) if self.phylogenetic_weights else 1.0
                    lang_candidates[lang] = (candidates, weight)

        if not lang_candidates:
            return None

        #Find inter-language consensus with weighted comparison
        common_candidates = set()
        lang_pairs = list(itertools.combinations(lang_candidates.items(), 2))

        for (lang1, (cands1, w1)), (lang2, (cands2, w2)) in lang_pairs:
            for cand1 in cands1:
                for cand2 in cands2:
                    if self.enhanced_similarity(cand1, cand2) > 0.8:
                        best = cand1 if w1 >= w2 else cand2
                        common_candidates.add(best)

        #Score candidates
        scored = []
        for candidate in (common_candidates or
                            set().union(*[cands for cands, _ in lang_candidates.values()])):
            scored.append((
                candidate,
                self._latin_likeness_score(candidate, observed_forms)
            ))

        if scored:
            best_candidate, best_score = max(scored, key=lambda x: x[1])
            return best_candidate if best_score > -5.0 else None

        return None

    def _filter_by_form_likeness(self, candidates, max_size):
        "filter forms by highest likeness score"
        scored = []
        for cand in candidates:
            score = self._latin_likeness_score(cand)
            scored.append((cand, score))

        scored.sort(key=lambda x: -x[1])
        return {cand for cand, _ in scored[:max_size]}

    def _latin_likeness_score(self, candidate, observed_forms=None):
        """Calculate how Latin-like a word form is"""
        if not candidate:
            return -10.0

        score = 0.0
        
        for c in candidate:
            score += self.latin_phoneme_freq.get(c, -0.1)

        # Length Normalization
        if observed_forms:
            lengths = [len(w) for w in observed_forms.values() if w and w != '-']
            if lengths:
                mean_len = (sum(lengths) / len(lengths)) + 2
                len_diff = abs(len(candidate) - mean_len)
                if len_diff > 2:
                    score -= 0.3 * (len_diff - 2)

        # Absolute Length Boundaries
        if len(candidate) < 3:
            score -= 2.0
        elif len(candidate) > 12:
            score -= 0.4 * (len(candidate) - 12)

        # Morphological Patterns
        ending_bonus = 0
        for ending in self.common_endings:
            if candidate.endswith(ending):
                ending_bonus = 0.7 if ending in self.common_endings else 0.2
                break
        score += ending_bonus

        for prefix in self.common_prefixes:
            if candidate.startswith(prefix):
                score += 0.5
                break

        # Phonotactic Constraints
        if not any(c in self.vowels for c in candidate):
            score -= 6.0  # Severe penalty for words without vowels

        if any(c in self.invalid_sequences for c in candidate ):
            score -= 1.0 #penalty for words with invalid consonant clusters

        # Consonant clusters
        consecutive_consonants = 0
        for c in candidate:
            if c in self.consonants:
                consecutive_consonants += 1
                if consecutive_consonants > 3:
                    score -= 0.6
            else:
                consecutive_consonants = 0

        # Vowel harmony
        vowel_sequence = ''.join(c for c in candidate if c in self.vowels)
        for pattern in self.vowel_patterns:
            v1, v2 = pattern.split('-')
            if v1 in vowel_sequence and v2 in vowel_sequence:
                score += 0.2
                break

        # Valid consonant clusters
        has_valid_cluster = False
        for cluster in self.valid_consonant_clusters:
            if cluster in candidate:
                has_valid_cluster = True
                score += 0.5
                break

        return score

    def enhanced_similarity(self, word1, word2):
        """Calculate enhanced similarity between two word forms"""
        if not word1 or not word2:
            return 0.0

        # Compute normalized similarity
        m, n = len(word1), len(word2)
        max_len = max(m, n)
        distance = levenshtein_distance(word1, word2)
        similarity = 1.0 - (distance / max_len)

        # Apply phonological class bonus
        class_bonus = 0.0
        min_len = min(m, n)
        for i in range(min_len):
            c1, c2 = word1[i], word2[i]
            if c1 != c2:
                if (c1 in self.vowels and c2 in self.vowels) or \
                   (c1 in self.stops and c2 in self.stops) or \
                   (c1 in self.fricatives and c2 in self.fricatives) or \
                   (c1 in self.liquids and c2 in self.liquids) or \
                   (c1 in self.nasals and c2 in self.nasals):
                    class_bonus += 0.05

        return min(1.0, similarity + class_bonus)

    def prior(self, protoform):
        """Calculate prior probability of a protoform based on training data"""
        if not protoform:
            return float('-inf')

        # Use frequencies from training data if available
        prior_score = math.log(self.protoform_counter.get(protoform, 0.1) + 0.1)

        # Add morphological plausibility
        latin_likeness = self._latin_likeness_score(protoform)

        return prior_score + (0.5 * latin_likeness)

    def likelihood(self, protoform, observed_forms):
        """Calculate likelihood of observed forms given a protoform"""
        if not protoform or not observed_forms:
            return float('-inf')

        total_likelihood = 0.0

        for lang, word in observed_forms.items():
            if not word or word == '-':
                continue

            # Generate expected form by applying rules
            expected = self.apply_rules(protoform, lang)

            if not expected:
                continue

            # Calculate similarity between expected and observed
            sim = self.enhanced_similarity(expected, word)
            if sim > 0:
                likelihood = math.log(sim)
            else:
                likelihood = -5.0  # Penalty for zero similarity

            # Apply phylogenetic weighting 
            if self.phylogenetic_weights:
                weight = self.phylogenetic_weights.get(lang, 1.0)
                likelihood *= weight

            total_likelihood += likelihood

        return total_likelihood

    def reconstruct_cognate_set(self, observed_forms, num_candidates=10):
        """Reconstruct Latin protoform for a set of cognates"""
       
        consensus = self.get_consensus_form(observed_forms)

        if not consensus:
            non_empty = [(lang, word) for lang, word in observed_forms.items() if word and word != '-']
            if non_empty:
                if self.phylogenetic_weights:
                    lang, word = max(non_empty, key=lambda x: self.phylogenetic_weights.get(x[0], 0))
                else:
                    lang, word = non_empty[0]

                initial_candidates = self.reverse_apply_rules(word, lang, observed_forms)
                if initial_candidates:
                    consensus = next(iter(initial_candidates))
                else:
                    return []
            else:
                return []

        candidates = {consensus}
        for _ in range(2): 
            new_candidates = set()
            for cand in candidates:
                new_candidates.update(self.generate_variants(cand))
            candidates.update(new_candidates)

        #Run beam search to improve candidates
        initial_protoforms = list(candidates)[:20]  # Limit to top 10 initial candidates
        beam_results = self.beam_search(initial_protoforms, observed_forms, beam_width=5, max_iter=5)

        return [proto for proto, _ in beam_results[:num_candidates]]

    def analyze_row(self, row):
          """
          Process a single row of data using a Darwinian selection approach
          to find the optimal Latin reconstruction.
          """
          observed_forms = {
              lang.lower(): row.get(lang.upper(), '-')
              for lang in self.languages
          }

          valid_forms = {lang: word for lang, word in observed_forms.items() if word and word != '-'}
          if not valid_forms:
              return None

          #Initialize the pool of candidates
          current_candidates = set()

          # Add Seed top candidates from phase I, constructed top rule-transformations to be added to pool
          for i in range(1, 6):
            for lang, word in valid_forms.items():
                proto_key = f'Selected_Proto{i}'
                if proto_key in row and row[proto_key] and row[proto_key] != '-':
                    current_candidates.add(row[proto_key])
                    transformed_proto=self.reverse_apply_rules(row[proto_key], lang)
                    transformed_protos=[]
                    for item in transformed_proto:
                      transformed_protos.append((item, self.score_protoform(item,observed_forms)))
                    transformed_protos.sort(key=lambda x: -x[1])
                    current_candidates.update([cand for cand, _ in transformed_protos[:1]]) # take top transformed


          # # Add transformed forms from each reflex (take top 1-2 from each language's transformation) to test Ranked-Prob-Evo-Ext
          # for lang, word in valid_forms.items():
          #     transformed_cands = self.reverse_apply_rules(word, lang)
          #     # Take a few top-scoring transformed candidates from each language
          #     scored_transformed_cands = []
          #     for cand in transformed_cands:
          #         # Use a temporary score for initial selection, refined later
          #         scored_transformed_cands.append((cand, self.score_protoform(cand, observed_forms)))
          #     scored_transformed_cands.sort(key=lambda x: -x[1])
          #     current_candidates.update([cand for cand, _ in scored_transformed_cands[:2]]) 

   
          current_candidates = {cand for cand in current_candidates if cand and cand != '-'}

          if not current_candidates:
              # Fallback if no initial candidates are found
              candidates = self.reconstruct_cognate_set(observed_forms, num_candidates=2)
              return candidates[0] if candidates else None

          #Darwinian Selection Loop
          num_eliminated_per_round = max(1, len(current_candidates) // 5) # Eliminate low scoring candidates
          max_selection_rounds = 20 

          for round_num in range(max_selection_rounds):
              if len(current_candidates) <= 1:
                  break # Stop if only one or zero candidates remain

              # Calculate average length of current surviving candidates
              candidate_lengths = [len(c) for c in current_candidates]
              if candidate_lengths:
                  avg_length = sum(candidate_lengths) / len(candidate_lengths)
                  min_acceptable_length = avg_length - 2
                  max_acceptable_length = avg_length + 4
              else:
                  avg_length = 0 
                  min_acceptable_length = 0
                  max_acceptable_length = float('inf')


              # Score all current candidates with dynamic length penalty
              scored_candidates = []
              for cand in current_candidates:
                  base_score = self.score_protoform(cand, observed_forms)

                  length_penalty = 0.0
                  cand_len = len(cand)
                  if cand_len < min_acceptable_length:
                      length_penalty = (min_acceptable_length - cand_len) * 0.8 # Stronger penalty for being too short
                  elif cand_len > max_acceptable_length:
                      length_penalty = (cand_len - max_acceptable_length) * 0.5 # Slightly less severe for too long

                  final_score = base_score - length_penalty
                  scored_candidates.append((cand, final_score))

              scored_candidates.sort(key=lambda x: -x[1])

              # Eliminate the lowest-scoring candidates
              num_to_keep = max(1, len(scored_candidates) - num_eliminated_per_round)
              current_candidates = {cand for cand, _ in scored_candidates[:num_to_keep]}

              # Introduce new variants from the top candidates to simulate "mutation"
              if len(current_candidates) < 5 and round_num < max_selection_rounds - 1: 
                  new_variants_generated = set()
                  for proto in list(current_candidates)[:min(len(current_candidates), 4)]: 
                      new_variants_generated.update(self.generate_variants(proto))
                  current_candidates.update(new_variants_generated)
                  current_candidates = {cand for cand in current_candidates if cand and cand != '-'} 

          # The last surviving protoform is the result
          if current_candidates:
              final_candidates_scored = []
              for cand in current_candidates:
                  final_candidates_scored.append((cand, self.score_protoform(cand, observed_forms)))
              final_candidates_scored.sort(key=lambda x: -x[1])
              return final_candidates_scored[0][0]
          else:
              return None

    def get_transformed_forms(self, row):
      """
      Get transformed forms for each word in the row without performing reconstruction.
      Returns a list of dictionaries with word, language, and transformations.
      """
      observed_forms = {
          lang.lower(): row.get(lang.upper(), '-')
          for lang in self.languages
      }

      valid_forms = {lang: word for lang, word in observed_forms.items() if word and word != '-'}
      if not valid_forms:
          return []

      transformed_forms = []

      for lang, word in valid_forms.items():
          transformed_cands = self.reverse_apply_rules(word, lang)

          scored_transformed_cands = []
          for cand in transformed_cands:
              scored_transformed_cands.append((cand, self.score_protoform(cand, observed_forms)))
          scored_transformed_cands.sort(key=lambda x: -x[1])

          top_transforms = scored_transformed_cands[:2]

          for i, (cand, score) in enumerate(top_transforms, 1):
              transformed_forms.append({
                  'word': word,
                  'language': lang,
                  f'top_{i}_transformation': cand,
                  'score': score
              })

      return transformed_forms
