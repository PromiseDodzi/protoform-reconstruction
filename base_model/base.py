#python base.py.py -n "my_run" -m Dirichlet
import numpy as np
import torch
import re
import argparse
import time
import json
import os
from tqdm import tqdm
from collections import defaultdict

# Constants
BIG_NEG = -1e10
VOWELS = {'a', 'e', 'i', 'o', 'u', 'ɛ', 'ɔ', 'æ', 'ɪ', 'ʊ', 'ʌ', 'ə', 'ɨ', 'ɐ', 'ɑ', 'ɒ'} # Add all vowels from your IPA set
CONSONANTS = set()

# Data preprocessing functions
def remove_stress(word):
    word = word.replace('ˈ', '')
    word = word.replace('ˌ', '')
    return word

def remove_length_indicators(word):
    word = word.replace(':', '')
    word = word.replace('ː', '')
    word = word.replace('-', '')
    return word 

def remove_parentheses_labels(word):
    return re.sub(r"[\(\[](.*?)[\)\]]", "", word)  # Added raw string prefix

def remove_superscripts(word):
    word = word.replace('ʲ', '')
    word = word.replace('ʰ', '')
    word = word.replace('ɔ̃'[1], '')
    return word 

def preprocess_ipa(words):
    processed = []
    for word in words:
        w = remove_stress(word)
        w = remove_length_indicators(w)
        w = remove_parentheses_labels(w)
        w = remove_superscripts(w)
        processed.append(w)
    return processed

def read_words(filename):
    with open(f'base_model/{filename}', 'r', encoding='utf-8') as f:
        words = f.readlines()
        words = [w.strip() for w in words]
    return words

def create_files(link='base_model/romance-ipa.txt'):
    with open(link, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    header = lines[0].split('\t')
    languages = header  # Include all columns as languages
    language_data = {lang: [] for lang in languages}

    for line in lines[1:]:
        if not line.strip():
            continue
        
        parts = line.split('\t')
        
        # Process each language's data correctly
        for i, lang in enumerate(languages):
            if i < len(parts):
                ipa = parts[i]
                # Only exclude '+' values, but include '-' values as empty strings
                if ipa.strip() == '-':
                    language_data[lang].append('')
                elif ipa.strip() != '+':
                    language_data[lang].append(ipa)
                # Skip '+' entries completely
            else:
                # If data is missing for this language, add empty string
                language_data[lang].append('')

    # Write each language's data to a separate file
    for lang, data in language_data.items():
        filename = f"base_model/{lang.capitalize().replace('-', '_')}_ipa.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data))
    
    print(f"Files created successfully! Languages processed: {', '.join(languages)}")
    # Optional: print counts for verification
    for lang, data in language_data.items():
        print(f"{lang}: {len(data)} entries")

def small_dataset(langs, per_n):
    fr, pt, ro, es, it, la = langs
    subset_indices = list(range(0, len(la), per_n))
    fr_sm = [fr[i] for i in subset_indices]
    pt_sm = [pt[i] for i in subset_indices]
    ro_sm = [ro[i] for i in subset_indices]
    es_sm = [es[i] for i in subset_indices]
    it_sm = [it[i] for i in subset_indices]
    la_sm = [la[i] for i in subset_indices]
    langs_sm = (fr_sm, pt_sm, ro_sm, es_sm, it_sm, la_sm)
    cogs_sm = [[lang[i] for lang in langs_sm] for i in range(len(fr_sm))]
    return langs_sm, cogs_sm

# Vocabulary class
class Vocabulary:
    def __init__(self, words):
        vocab = set()
        for word in words:
            for x in word:
                vocab.add(x)
        vocab = sorted(list(vocab))
        vocab.append('-')
        vocab.append('|')
        vocab.append('(')
        vocab.append(')')
        self.vocab = vocab
        self.vocab2id = {x: i for i, x in enumerate(vocab)}
        self.pad = self.vocab2id['-']
        self.step = self.vocab2id['|']
        self.dlt = self.step
        self.sow = self.vocab2id['(']
        self.eow = self.vocab2id[')']
        self.size = len(vocab)
        
        # Add fallback for unknown characters
        self.unk = self.pad
        self.vocab2id = defaultdict(lambda: self.unk, self.vocab2id)

    def string2ids(self, s):
        return [self.vocab2id[x] for x in s]
    
    def ids2string(self, ids):
        s = ''.join([self.vocab[i] for i in ids])
        s = s.replace('-', '')
        s = s.replace('#', '')
        return s

    def make_batch(self, words, align_right=False):
        # Ensure words is a list of strings
        words = [str(w) if not isinstance(w, str) else w for w in words]
        
        lens = [len(w) for w in words]
        max_len = max(lens) if lens else 0
        batch = np.full((max_len, len(words)), fill_value=self.pad, dtype=np.int32)
        
        for i, w in enumerate(words):
            for j, x in enumerate(str(w)):
                try:
                    if align_right:
                        batch[max_len - len(w) + j, i] = self.vocab2id[x]
                    else:
                        batch[j, i] = self.vocab2id[x]
                except:
                    batch[j, i] = self.unk
                    
        return batch, np.array(lens)

    def make_tensor(self, words, add_boundaries=False):
        if isinstance(words, (np.ndarray, torch.Tensor)):
            if words.ndim == 1:
                words = np.expand_dims(words, 1)
            return torch.from_numpy(words.astype(np.int64))
            
        if add_boundaries:
            words = ['(%s)'%w for w in words]
            
        np_batch, lens = self.make_batch(words)
        return torch.from_numpy(np_batch.astype(np.int64))

# Models
class ProbCache:
    def __init__(self, sources, targets, fill=BIG_NEG):
        batch_size = sources.shape[1]
        self.sub = np.full((len(sources), len(targets), batch_size), fill_value=fill)
        self.dlt = np.full((len(sources), len(targets), batch_size), fill_value=fill)
        self.ins = np.full((len(sources), len(targets), batch_size), fill_value=fill)
        self.end = np.full((len(sources), len(targets), batch_size), fill_value=fill)

class BaseModel:
    def sub(self, source, targets, i, j):
        return NotImplemented
    def dlt(self, source, targets, i, j):
        return NotImplemented
    def ins(self, source, targets, i, j):
        return NotImplemented
    def end(self, source, targets, i, j):
        return NotImplemented
    
    def cache_probs(self, sources, targets):
        return NotImplemented
    def m_step(self, sources, targets, posterior_cache):
        return NotImplemented

class DirichletModel(BaseModel):
    def __init__(self, initial_prs, natural_classes):
        self_sub, dlt, end_ins = initial_prs
        self.num_cls = max(natural_classes) + 1
        self.cls_of = natural_classes
        
        sub_pr = (1.0 - dlt - self_sub) / vocab.size
        self.sub_prs = np.full((self.num_cls, vocab.size, vocab.size), fill_value=sub_pr)
        self.sub_prs[:, range(vocab.size), range(vocab.size)] = self_sub
        self.sub_prs[:, :, vocab.step] = dlt
        
        ins_pr = (1.0 - end_ins) / vocab.size
        self.ins_prs = np.full((self.num_cls, vocab.size, vocab.size), fill_value=ins_pr)
        self.ins_prs[:, :, vocab.step] = end_ins

        self.normalize()
        self.init_sub_prs = np.exp(self.sub_prs)
        self.init_ins_prs = np.exp(self.ins_prs)
        self.aligner = self 

    def normalize(self):
        # Add small epsilon to avoid log(0)
        self.sub_prs = self.sub_prs + 1e-10
        self.ins_prs = self.ins_prs + 1e-10
        
        # Normalize and take log
        self.sub_prs = self.sub_prs / (self.sub_prs.sum(axis=-1, keepdims=True) + 1e-10)
        self.sub_prs = np.log(self.sub_prs)
        
        self.ins_prs = self.ins_prs / (self.ins_prs.sum(axis=-1, keepdims=True) + 1e-10)
        self.ins_prs = np.log(self.ins_prs)
        
        # Replace any remaining invalid values with large negative numbers
        self.sub_prs[np.isnan(self.sub_prs)] = BIG_NEG
        self.ins_prs[np.isnan(self.ins_prs)] = BIG_NEG

    def __sub_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[targets[j]] 
        return self.sub_prs[left_context, input_char, targets[j+1]]

    def __dlt_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[targets[j]] 
        return self.sub_prs[left_context, input_char, vocab.dlt]
    
    def __ins_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[targets[j]]
        return self.ins_prs[left_context, input_char, targets[j+1]] 

    def __end_pr(self, sources, targets, i, j):
        input_char = sources[i]
        left_context = self.cls_of[targets[j]]
        return self.ins_prs[left_context, input_char, vocab.dlt] 
                
    def cache_probs(self, sources, targets):
        batch_size = sources.shape[1]
        self.cache = ProbCache(sources, targets)

        for i in range(len(sources)):
            for j in range(len(targets)):
                if j+1 < len(targets):
                    self.cache.sub[i, j] = self.__sub_pr(sources, targets, i, j) 
                    self.cache.ins[i, j] = self.__ins_pr(sources, targets, i, j)
                
                self.cache.dlt[i, j] = self.__dlt_pr(sources, targets, i, j)
                self.cache.end[i, j] = self.__end_pr(sources, targets, i, j)
                
    def sub(self, i, j):
        return self.cache.sub[i, j]
    
    def ins(self, i, j):
        return self.cache.ins[i, j]

    def dlt(self, i, j):
        return self.cache.dlt[i, j]

    def end(self, i, j):
        return self.cache.end[i, j]

    def m_step(self, sources, targets, posterior_cache, smooth=0.1):
        sources = vocab.make_tensor(sources, add_boundaries=True).numpy()
        targets = vocab.make_tensor(targets, add_boundaries=True).numpy() 
        batch_size = sources.shape[1]
        self.sub_prs = np.zeros((self.num_cls, vocab.size, vocab.size))
        self.ins_prs = np.zeros((self.num_cls, vocab.size, vocab.size))
        self.sub_prs = self.sub_prs + smooth * self.init_sub_prs
        self.ins_prs = self.ins_prs + smooth * self.init_ins_prs
        self.normalize()

        sub_post = np.exp(posterior_cache.sub)
        dlt_post = np.exp(posterior_cache.dlt)
        ins_post = np.exp(posterior_cache.ins)
        end_post = np.exp(posterior_cache.end)

        for i in range(len(sources)):
            for j in range(len(targets)):
                input_char = sources[i]
                left_context = self.cls_of[targets[j]]

                if j < len(targets)-1:
                    outcomes = targets[j+1] 
                    for n in range(batch_size):
                        self.sub_prs[left_context[n], input_char[n], outcomes[n]] += sub_post[i, j, n]
                        self.ins_prs[left_context[n], input_char[n], outcomes[n]] += ins_post[i, j, n]

                for n in range(batch_size):
                    self.sub_prs[left_context[n], input_char[n], vocab.dlt] += dlt_post[i, j, n]
                    self.ins_prs[left_context[n], input_char[n], vocab.dlt] += end_post[i, j, n]

        self.sub_prs = self.sub_prs + smooth * self.init_sub_prs
        self.ins_prs = self.ins_prs + smooth * self.init_ins_prs
        self.normalize()

class BigramModel:
    def __init__(self, words):
        self.counts = np.zeros((vocab.size, vocab.size)) + 1e-10  # Add smoothing
        for word in words:
            ids = vocab.string2ids(word)
            for i in range(len(ids)-1):
                self.counts[ids[i], ids[i+1]] += 1
        self.probs = self.counts / (self.counts.sum(axis=1, keepdims=True) + 1e-10)
        self.probs = np.log(self.probs)
    
    def string_logp(self, words):
        logps = []
        for word in words:
            ids = vocab.string2ids(word)
            logp = 0.0
            for i in range(len(ids)-1):
                logp += self.probs[ids[i], ids[i+1]]
            logps.append(logp)
        return np.array(logps)

MODELS = {
    'Dirichlet': DirichletModel,
    'Bigram': BigramModel
}

# Alignment and reconstruction functions
def compute_posteriors(aligner, sources, targets):
    # Ensure proper dimensionality
    if isinstance(sources, list):
        sources = vocab.make_tensor(sources, add_boundaries=True).numpy()
    if isinstance(targets, list):
        targets = vocab.make_tensor(targets, add_boundaries=True).numpy()
        
    if sources.ndim == 1:
        sources = np.expand_dims(sources, 1)
    if targets.ndim == 1:
        targets = np.expand_dims(targets, 1)
        
    aligner.cache_probs(sources, targets)
    return aligner.cache

def compute_mutation_prob(model, sources, targets):
    # Handle empty inputs
    if not sources or not targets:
        return np.array([BIG_NEG]*max(1, len(sources)))
        
    # Convert to lists if needed
    sources = list(sources)
    targets = list(targets)
    
    # Ensure targets matches sources length
    if len(targets) == 1 and len(sources) > 1:
        targets = targets * len(sources)
        
    logps = []
    for s, t in zip(sources, targets):
        try:
            posteriors = compute_posteriors(model.aligner, [s], [t])
            logp = posteriors.sub.sum() + posteriors.dlt.sum() + posteriors.ins.sum() + posteriors.end.sum()
            if np.isnan(logp):
                logp = BIG_NEG
            logps.append(logp)
        except:
            logps.append(BIG_NEG)
    
    logps = np.array(logps)
    logps[np.isnan(logps)] = BIG_NEG  # Ensure no NaN values remain
    
    return logps

def batch_edit_dist(s1, s2):
    dists = []
    for a, b in zip(s1, s2):
        dists.append(edit_distance(a, b))
    return np.array(dists)

def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def one_edit_proposals(word, leaves):
    proposals = set()
    proposals.add(word)  # Always include current word
    
    # Phoneme-aware edits
    for i in range(len(word)):
        current = word[i]
        # Vowel/consonant aware substitutions
        if current in VOWELS:
            replacements = VOWELS
        else:
            replacements = CONSONANTS
            
        for r in replacements:
            new_word = word[:i] + r + word[i+1:]
            if any(new_word in leaf for leaf in leaves):
                proposals.add(new_word)
    
    common_changes = {
    # Palatalization
    'k': 'ʧ',      # Latin "c" before front vowels (e.g., "centum" → "ciento")
    'g': 'ʤ',      # Latin "g" before front vowels (e.g., "gelu" → "gel")
    # Voicing of stops
    'p': 'b',
    't': 'd',
    'k': 'g',
    # Lenition (consonant weakening)
    'b': 'β',
    'd': 'ð',
    'g': 'ɣ',
    # Nasal assimilation and simplification
    'mn': 'n',     # "somnus" → "sommeil"
    'gn': 'ɲ',     # "signum" → "señal"
    'nn': 'n',
    # Cluster reduction
    'kt': 'tt',    # "nocte" → "notte"
    'ct': 'it',    # "factum" → "fait"
    'pt': 't',
    'bt': 't',
    # Glide insertion and vowel diphthongization
    'e': 'je',     # "bene" → "bien"
    'o': 'we',     # "bonus" → "bueno"
    'ae': 'e',     # "caelum" → "cielo"
    'au': 'o',     # "aurum" → "oro"
    # Simplifications
    'ti': 'ts',    # "natio" → "nación"
    'di': 'ʤ',     # "diurnus" → "giorno"
    # Final vowel loss
    'us': '',      # "lupus" → "loup"
    'um': '',      # "bellum" → "bel"
    # Vowel raising or lowering
    'ɛ': 'e',
    'ɔ': 'o',
    # Metathesis
    'per': 'pre',  # "periculum" → "peril"
    # Other known changes
    'll': 'ʎ',     # Italian "figlio", Spanish "hijo" from "filius"
    'fl': 'ʎ',     # Latin "flamma" → "llama"
    # Sibilant development
    's': 'z',      # In intervocalic position
}

    
    for pattern, replacement in common_changes.items():
        if pattern in word:
            new_word = word.replace(pattern, replacement)
            proposals.add(new_word)
    
    return list(proposals)

def heuristic_proposals(word, leaves):
    proposals = set()
    for leaf in leaves:
        if abs(len(word) - len(leaf)) <= 2:
            proposals.add(leaf)
    return list(proposals)

# Tree structure classes
class LanguageNode:
    def __init__(self, children, model, name, config):
        self.config = config
        self.children = children
        for child in self.children:
            child.top = self
        self.top = None 
        self.model = model 
        self.is_leaf = False
        self.is_root = False 
        self.name = name 
        self.initialize_samples()

    def initialize_samples(self):
        words = []
        for i in range(len(self.children[0].words)):
            leaves = [child.words[i] for child in self.children]
            words.append(np.random.choice(leaves))
        self.words = words

    def sample_recon(self, mh, i):
        cur = self.words[i]
        sample_likelihood = None
        num_rounds = 1 if bool(self.config['proposal_heuristic']) else 5
        
        # Handle empty initial form
        if not cur:
            leaves = [w for w in [child.words[i] for child in self.children] if w]
            cur = min(leaves, key=len) if leaves else ""  # Fallback to empty string if no leaves
            if not cur:  # Still empty after fallback
                return "", BIG_NEG

        for _ in range(num_rounds):
            leaves = [child.words[i] for child in self.children]
            
            # Generate proposals
            if bool(self.config['proposal_heuristic']):
                proposals = heuristic_proposals(cur, leaves)
            else:
                proposals = one_edit_proposals(cur, leaves)
            
            # Handle empty proposals case
            if not proposals:
                proposals= [cur]
            
            # Limit number of proposals if too many
            if len(proposals) > 500:
                proposals = np.random.choice(proposals, size=500, replace=False)
            
            proposal_prs = np.zeros(len(proposals))

            for child in self.children:
                leaf = child.words[i]
                if not bool(self.config['reverse_models']):
                    branch_pr = compute_mutation_prob(child.model, proposals, [leaf]*len(proposals))
                else:
                    branch_pr = compute_mutation_prob(child.model, [leaf]*len(proposals), proposals)                
                proposal_prs += branch_pr
            
            if self.is_root:
                lm_pr = self.lm.string_logp(proposals)
                proposal_prs += lm_pr 
            else:
                if not bool(self.config['reverse_models']):
                    top_pr = compute_mutation_prob(self.model, [self.top.words[i]]*len(proposals), proposals)
                else:
                    top_pr = compute_mutation_prob(self.model, proposals, [self.top.words[i]]*len(proposals))
                proposal_prs += top_pr 
            if mh:
                temp = self.config.get('temperature', 1.0)
                scaled_prs = (proposal_prs - proposal_prs.max()) / temp
                
                # Handle numerical stability
                prs = np.exp(scaled_prs - np.max(scaled_prs))  # Subtract max for numerical stability
                prs = prs / (prs.sum() + 1e-10)  # Add small epsilon to avoid division by zero
                
                # Replace any remaining NaN or invalid values with uniform probabilities
                if np.any(np.isnan(prs)) or np.any(prs < 0):
                    prs = np.ones_like(prs) / len(prs)
                    
                idx = np.random.choice(range(len(proposals)), p=prs)
                cur = proposals[idx]
                sample_likelihood = proposal_prs[idx]
            if not mh:
                argmax = np.argmax(proposal_prs)
                sample_likelihood = proposal_prs[argmax]
                if proposals[argmax] == cur:
                    break 
                cur = proposals[argmax]
            else:
                # Add temperature scaling if config exists
                temp = self.config.get('temperature', 1.0)
                scaled_prs = (proposal_prs - proposal_prs.max()) / temp
                prs = np.exp(scaled_prs) / np.exp(scaled_prs).sum()
                idx = np.random.choice(range(len(proposals)), p=prs)
                cur = proposals[idx]
                sample_likelihood = proposal_prs[idx]

        return cur, sample_likelihood

    def sample_reconstructions(self, mh):
        print('Sampling reconstructions for %s' % self.name)
        recons = []
        likelihood = 0.0

        for i in tqdm(range(len(self.words)), leave=False):
            sample, l = self.sample_recon(mh, i)
            recons.append(sample)
            likelihood += l 

        self.words = recons
        return likelihood

    def compute_likelihood(self, samples):
        log_prs = np.zeros(len(samples))
        for child in self.children:
            if not bool(self.config['reverse_models']):
                branch_pr = compute_mutation_prob(child.model, samples, child.words)
            else:
                branch_pr = compute_mutation_prob(child.model, child.words, samples)
            log_prs += branch_pr 
        if self.is_root:
            lm_pr = self.lm.string_logp(samples)
            log_prs += lm_pr
        return log_prs 

    def update_model(self, **kwargs):
        if self.is_root:
            self.lm = BigramModel(self.words)
        else:
            if not bool(self.config['reverse_models']):
                posteriors = compute_posteriors(self.model.aligner, self.top.words, self.words)
                self.model.m_step(self.top.words, self.words, posteriors, **kwargs)
            else:
                posteriors = compute_posteriors(self.model.aligner, self.words, self.top.words)
                self.model.m_step(self.words, self.top.words, posteriors, **kwargs)

class LeafNode(LanguageNode):
    def __init__(self, words, model, name, config):
        self.config = config
        self.words = words
        self.model = model 
        self.is_leaf = True
        self.is_root = False 
        self.name = name 
        self.top = None 

class RootNode(LanguageNode):
    def __init__(self, children, lm, name, config):
        self.config = config 
        self.children = children
        for child in self.children:
            child.top = self
        self.lm = lm 
        self.is_root = True
        self.is_leaf = False 
        self.name = name
        self.initialize_samples()

# EM algorithm
def sample_tree(node, mh):
    if node.is_leaf:
        return 0
    likelihood = 0
    for child in node.children:
        lc = sample_tree(child, mh)
        likelihood += lc
    likelihood += node.sample_reconstructions(mh=mh)
    return likelihood

def mstep_tree(node, **kwargs):
    if not node.is_leaf:
        for child in node.children:
            mstep_tree(child, **kwargs)
    if not node.is_root:
        node.update_model(**kwargs)

def evaluate(recons, la):
    return batch_edit_dist(recons, la).mean()

def run_EM(root, EM_ROUNDS, LOG_OUTPUT=False, LOG_DIR=None, latin_words=None):
    hist = []
    for itr in range(EM_ROUNDS):
        likelihood = sample_tree(root, mh=True)
        print("Likelihood:", likelihood)
        dist = evaluate(root.words, latin_words)  # Use the passed parameter
        print("Edit distance:", dist)
        hist.append(dist)
        
        if LOG_OUTPUT:
            with open(LOG_DIR + 'iteration%d.txt' % itr, 'w') as f:
                f.writelines([s+'\n' for s in root.words])
        mstep_tree(root)
    print("Distance history:", hist)
    return root

def main():
    parser = argparse.ArgumentParser(description='Reconstruct protoforms from Romance language data.')
    parser.add_argument('-r', '--reverse_models', action='store_true', default=False,
                       help='Use reverse direction models')
    parser.add_argument('-e', '--em_rounds', default=10, type=int,
                       help='Number of EM iterations to run')
    parser.add_argument('-p', '--proposal_heuristic', action='store_true', default=False,
                       help='Use heuristic proposal method')
    parser.add_argument('-n', '--name', required=True,
                       help='Name for this run')
    parser.add_argument('-c', '--conditioning_type', default=0, type=int,
                       help='Type of natural class conditioning (0-2)')
    parser.add_argument('-l', '--lm_percent', default=1.0, type=float,
                       help='Percentage of Latin data to use for language model')
    parser.add_argument('-f', '--flat_tree', action='store_true', default=False,
                       help='Use flat tree structure')
    parser.add_argument('-s', '--small_dataset', type=int, default=None,
                       help='Use smaller dataset (specify step size)')
    parser.add_argument('-k', '--use_cluster_model', action='store_true', default=False,
                       help='Use cluster model (not implemented)')
    parser.add_argument('-d', '--distance_threshold', default=10.0, type=float,
                       help='Distance threshold (not implemented)')
    parser.add_argument('-o', '--log_output', action='store_true', default=False,
                       help='Save output logs')
    parser.add_argument('-m', '--model', required=True, choices=['Dirichlet', 'Bigram'],
                       help='Model type to use')
    parser.add_argument('-i', '--input_file', default='romance-ipa.txt',
                       help='Input data file')

    args = parser.parse_args()
    config = vars(args)
    print("Configuration:", config)

    RUN_NAME = args.name 
    LOG_DIR = 'log/' + RUN_NAME + '_' + str(int(time.perf_counter())) + '/'
    EM_ROUNDS = int(args.em_rounds)
    LM_PERCENT = float(args.lm_percent)
    CONDITIONING_TYPE = int(args.conditioning_type)
    USE_FLAT_TREE = bool(args.flat_tree)
    LOG_OUTPUT = bool(args.log_output)
    MODEL_NAME = args.model

    # Create data files from input
    create_files(args.input_file)

    config['temperature'] = 0.1

    # Load and preprocess data
    fr = preprocess_ipa(read_words('French_ipa.txt'))
    pt = preprocess_ipa(read_words('Portuguese_ipa.txt'))
    ro = preprocess_ipa(read_words('Romanian_ipa.txt'))
    es = preprocess_ipa(read_words('Spanish_ipa.txt'))
    it = preprocess_ipa(read_words('Italian_ipa.txt'))
    la = preprocess_ipa(read_words('Latin_correct_ipa.txt'))
    langs = (fr, pt, ro, es, it, la)
    all_la = la.copy()

   
    # Initialize vocabulary with all words
    global vocab
    vocab = Vocabulary(fr+pt+ro+es+it+la)
    
    global VOWELS, CONSONANTS
    # Update the CONSONANTS set based on the actual vocabulary
    CONSONANTS = set(vocab.vocab) - VOWELS - {'-', '|', '(', ')'}

    VOW_CON_CLASSES = np.zeros(vocab.size, dtype=np.int32)
    for i, char in enumerate(vocab.vocab):
        VOW_CON_CLASSES[i] = 0 if char in VOWELS else 1
     # Natural classes definitions
    DEGEN_CLASSES = np.zeros(vocab.size, dtype=np.int32)
    # VOW_CON_CLASSES = np.zeros(vocab.size, dtype=np.int32)
    IDENTITY_CLASSES = np.array(range(vocab.size), dtype=np.int32)

    # Set natural classes based on conditioning type
    natural_classes = [DEGEN_CLASSES, VOW_CON_CLASSES, IDENTITY_CLASSES][CONDITIONING_TYPE]

    if args.small_dataset is not None:
        langs, cogs = small_dataset(langs, int(args.small_dataset))
        fr, pt, ro, es, it, la = langs

    if LOG_OUTPUT:
        print('Writing log into', LOG_DIR)
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_DIR + 'config.json', 'w') as f:
            json.dump(config, f)

    # Create language model
    lm_data = all_la.copy()
    np.random.shuffle(lm_data)
    lm_data = lm_data[:int(len(lm_data) * LM_PERCENT)]
    lm = BigramModel(lm_data)

    INITIAL_PRS = (0.7, 0.1, 0.2)
    
    ModelClass = MODELS[MODEL_NAME]
    if MODEL_NAME == 'Dirichlet':
        M = lambda : ModelClass(INITIAL_PRS, natural_classes)
    else:
        raise Exception("Unsupported model type")
    
    # Create leaf nodes for each language
    italian = LeafNode(it, M(), 'IT', config)
    spanish = LeafNode(es, M(), 'ES', config)
    portuguese = LeafNode(pt, M(), 'PT', config)
    french = LeafNode(fr, M(), 'FR', config)
    romanian = LeafNode(ro, M(), 'RO', config)

    # Create tree structure
    if USE_FLAT_TREE:
        root = RootNode([spanish, italian, portuguese, french, romanian], lm, 'LA', config)
    else:
        ibero = LanguageNode([spanish, portuguese], M(), 'IBERO', config)
        western = LanguageNode([ibero, french], M(), 'WESTERN', config)
        italo = LanguageNode([western, italian], M(), 'ITALO', config)
        root = RootNode([italo, romanian], lm, 'LA', config)

    # Run EM algorithm
    root = run_EM(root, EM_ROUNDS, LOG_OUTPUT, LOG_DIR, latin_words=la)

    # Final sampling
    sample_tree(root, mh=False)
    if LOG_OUTPUT:
        print('Final recons ', evaluate(root.words, la))
        with open(LOG_DIR + 'final_recons.txt', 'w') as f:
            f.writelines([s+'\n' for s in root.words])
    
    # Save the predicted protoforms
    predicted_protoforms = root.words
    with open('base_model/predicted_protoforms.txt', 'w', encoding='utf-8') as f:
        f.writelines([s+'\n' for s in predicted_protoforms])
    
    print("\nFirst 10 predicted protoforms:")
    for i, protoform in enumerate(predicted_protoforms[:10]):
        print(f"{i+1}. {protoform}")

    return predicted_protoforms

if __name__ == '__main__':
    predicted_protoforms = main()