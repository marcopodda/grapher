from learner.experiment import Experiment


def run():
    rundir = "/home/dottor/code/grapher/RUNS/PROTEINS_full_2019-06-11T14:31:47.370309"
    exp = Experiment.load("PROTEINS_full", rundir)
    exp.resume()
    # exp = Experiment("PROTEINS_full")
    # exp.run()


if __name__ == "__main__":
    run()
