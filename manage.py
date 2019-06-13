"""
Manage experiments.

Usage:
  manage.py train <dataset>
  manage.py resume <rundir>
  manage.py baseline <name> <dataset> --metric=<metric>
  manage.py (-h | --help)

Options:
  -d --dataset DATASET    Path to a hyperparameter config file [default: MUTAG].
  -r --rundir PATH        Use specific run.
  -l --last               Use last run.
  -h --help               Show this screen.
"""

from docopt import docopt
from learner.experiment import Experiment, BaselineExperiment


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


if __name__ == "__main__":
    main()
