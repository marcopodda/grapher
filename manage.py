"""
Manage experiments.

Usage:
    manage.py train <model>
    manage.py order
    manage.py evaluate
    manage.py evaluate_order
    manage.py (-h | --help)

Options:
    -h --help               Show this screen.
"""

from docopt import docopt
from learner import get_exp_class
from learner.evaluator import Evaluator, OrderEvaluator
from utils.constants import DATASET_NAMES


def main():
    args = docopt(__doc__, help=True, version=None)

    if args["train"]:
        exp_class = get_exp_class(args['<model>'])
        for dataset in DATASET_NAMES:
            exp = exp_class(dataset)
            exp.train()
    elif args["order"]:
        exp_class = get_exp_class("ORDER")
        for order in ["bfs", "random", "smiles"]:
            for dataset in DATASET_NAMES:
                if order == "smiles" and dataset not in ["ENZYMES", "PROTEINS_full"]:
                    continue
                exp = exp_class(order, dataset)
                exp.train()
    elif args["evaluate"]:
        ev = Evaluator()
        ev.evaluate()
    elif args["evaluate_order"]:
        ev = OrderEvaluator()
        ev.evaluate()


if __name__ == "__main__":
    main()
