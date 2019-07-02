"""
Manage experiments.

Usage:
    manage.py train <model>
    manage.py evaluate <metric> --order
    manage.py (-h | --help)

Options:
    -h --help               Show this screen.
    -o --order              Evaluate ordering.
"""

from docopt import docopt
from learner import get_exp_class
from learner.evaluator import Evaluator, OrderEvaluator
from utils.constants import DATASET_NAMES, ORDER_NAMES


def main():
    args = docopt(__doc__, help=True, version=None)

    if args["train"]:
        if args['<model>'] == 'order':
            exp_class = get_exp_class("ORDER")
            for order in ORDER_NAMES:
                for dataset in DATASET_NAMES:
                    if order == "smiles" and dataset not in ["ENZYMES", "PROTEINS_full"]:
                        continue
                    exp = exp_class(order, dataset)
                    exp.train()
        else:
            exp_class = get_exp_class(args['<model>'])
            for dataset in DATASET_NAMES:
                exp = exp_class(dataset)
                exp.train()
    elif args["evaluate"]:
        if args['--order']:
            ev = OrderEvaluator(args['<metric>'])
        else:
            ev = Evaluator(args['<metric>'])
        ev.evaluate()



if __name__ == "__main__":
    main()
