# Protoform Reconstruction

This repository accompanies the paper **"Unsupervised Protoform Reconstruction through Parsimonious Rule-guided Heuristics and Evolutionary Search"** by Promise Dodzi Kpoglu. The repository contains the complete experimental framework, including datasets, source code, and evaluation scripts used to validate the proposed methodology.

## Repository Structure

The repository is organized into three main directories:

- **`ranked_prob_evo/`**: Implementation and results for the proposed model  
- **`base_model/`**: Implementation and results for the baseline comparison model  
- **`adhoc_files/`**: Auxiliary scripts and visualization tools for analysis  

## File Descriptions

### Primary Model (`ranked_prob_evo/`)

| File | Description |
|------|-------------|
| `preprocessing.py` | Data preprocessing pipeline for input normalization |
| `parsimony.py` | Implementation of the parsimony-based reconstruction component |
| `rule_transform_and_evolution.py` | Rule transformation and evolutionary search algorithms |
| `main.py` | Main execution script coordinating all model components |
| `evaluator.py` | Performance evaluation and metrics computation |
| `romance-ipa.txt` | Raw Romance language dataset in IPA notation |
| `romance-cleaned.txt` | Preprocessed and normalized dataset |
| `parsimony_reconstructions.tsv` | Top-k parsimonious reconstruction candidates |
| `selected_proto_results.tsv` | Ranked reconstruction outputs |
| `final_results.tsv` | Final protoform reconstruction proposals |
| `evaluation_results_adopted.tsv` | Quantitative performance metrics |

### Baseline Model (`base_model/`)

| File | Description |
|------|-------------|
| `base.py` | Reimplementation of **Bouchard et al. (2007)** baseline model |
| `base_evaluation.py` | Baseline model performance evaluation |
| `romance-ipa.txt` | Original Romance language dataset |
| `romance_cleaned.txt` | Preprocessed baseline dataset |
| `French_ipa.txt` | French language wordlist |
| `Italian_ipa.txt` | Italian language wordlist |
| `Spanish_ipa.txt` | Spanish language wordlist |
| `Portuguese_ipa.txt` | Portuguese language wordlist |
| `Romanian_ipa.txt` | Romanian language wordlist |
| `Latin_correct_ipa.txt` | Ground truth Latin protoforms for evaluation |
| `predicted_protoforms.txt` | Baseline model reconstruction outputs |
| `merged_output.tsv` | Consolidated baseline results and data |
| `evaluation_base_model.tsv` | Baseline model performance metrics |

### Analysis Tools (`adhoc_files/`)

| File | Description |
|------|-------------|
| `get_illustration.py` | Performance visualization generator |
| `flowchart.tex` | LaTeX source for model architecture diagram |
| `flowchart.pdf` | Model architecture and component visualization |
| `normalized_metrics.png` | Performance vs. rule complexity analysis |
| `rule_performance.csv` | Rule-set performance evaluation data |
| `extended_results.csv` | Comprehensive performance analysis |
| `rule_transformed_forms.csv` | Rule transformation outputs and rankings |
| `Makefile` | Build automation for documentation generation |

## Experimental Reproduction

### Prerequisites

1. **Clone the repository:**
   ```
   git clone [[repository-url](https://github.com/PromiseDodzi/protoform-reconstruction)]
   cd protoform-reconstruction

2. **Install dependencies:**
   ```
   pip install -r requirements.txt

3. **Ensure you are on the master branch::**
   ```
   git checkout master

