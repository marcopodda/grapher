"""
Manage experiments.

Usage:
  manage.py train <dataset>
  manage.py resume <rundir>
  manage.py evaluate <rundir>
  manage.py (-h | --help)

Options:
  -d --dataset DATASET    Path to a hyperparameter config file [default: MUTAG].
  -r --rundir PATH        Use specific run.
  -l --last               Use last run.
  -h --help               Show this screen.
"""

from docopt import docopt
from learner.experiment import Experiment


def main():
    args = docopt(__doc__, help=True, version=None)

    if args["train"]:
        exp = Experiment(args["<dataset>"])
        exp.train()
    elif args["resume"]:
        exp = Experiment.load(args["<rundir>"])
        exp.resume()
    elif args["evaluate"]:
        exp = Experiment.load(args["<rundir>"])
        exp.evaluate()


if __name__ == "__main__":
    main()
