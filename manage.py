"""
Manage experiments.

Usage:
    manage.py train <model>
    manage.py evaluate [--model MODEL] [--order]
    manage.py (-h | --help)

Options:
    -h --help               Show this screen.
    -o --order              Evaluate ordering.
    -m --model              Model name.
"""

from docopt import docopt
from experiment import get_exp_class, load_experiment
from experiment.evaluator import Evaluator, OrderEvaluator
from utils.constants import DATASET_NAMES, ORDER_NAMES, MODEL_NAMES


def main():
    args = docopt(__doc__, help=True, version=None)

    if args["train"]:
        if args['<model>'] == 'order':
            exp_class = get_exp_class("ORDER")
            for order in ORDER_NAMES:
                for dataset in DATASET_NAMES:
                    if order == "smiles" and dataset not in ["ENZYMES", "PROTEINS_full"]:
                        continue
                    try:
                        exp = exp_class(order, dataset)
                        exp.train()
                    except FileExistsError:
                        continue
        else:
            exp_class = get_exp_class(args['<model>'])
            for dataset in DATASET_NAMES:
                print(f"training dataset {dataset}")
                try:
                    exp = exp_class(dataset)
                    exp.train()
                except FileExistsError:
                    continue
    elif args["evaluate"]:
        if args['--order']:
            ev_class = OrderEvaluator
            for order in ORDER_NAMES:
                ev = OrderEvaluator(order)
                ev.evaluate()
            return
        if args['--model']:
            ev = Evaluator(args['MODEL'])
            ev.evaluate()
        else:
            for model in MODEL_NAMES:
                ev = ev_class(model)
                ev.evaluate()



if __name__ == "__main__":
    main()
