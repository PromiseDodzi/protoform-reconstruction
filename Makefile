.PHONY: all main evaluator base base_evaluation

all: main evaluator base base_evaluation

main:
	python -m ranked_prob_evo.main

evaluator:
	python -m ranked_prob_evo.evaluator

base:
	python base_model/base.py -n "test_run" -m Dirichlet --input_file ranked_prob_evo/romance-ipa.txt

base_evaluation:
	python -m base_model.base_evaluation



