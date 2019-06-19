"""
Manage experiments.

Usage:
    manage.py train <dataset>
    manage.py resume <rundir>
    manage.py baseline <name> <dataset> --metric=<metric>
    manage.py graphrnn <dataset>
    manage.py evaluate
    manage.py (-h | --help)

Options:
    -h --help               Show this screen.
"""

from docopt import docopt
from learner.experiment import Experiment, BaselineExperiment, GraphRNNExperiment
from learner.evaluator import Evaluator


def main():
    args = docopt(__doc__, help=True, version=None)

    if args["train"]:
        exp = Experiment(args["<dataset>"])
        exp.train()
    elif args["resume"]:
        exp = Experiment.load(args["<rundir>"])
        exp.resume()
    elif args["baseline"]:
        exp = BaselineExperiment(args['<name>'], args['--metric'], args['<dataset>'])
        exp.train()
    elif args["graphrnn"]:
        exp = GraphRNNExperiment(args['<dataset>'])
        exp.train()
    elif args["evaluate"]:
        ev = Evaluator()
        print(ev.evaluate())


if __name__ == "__main__":
    main()
