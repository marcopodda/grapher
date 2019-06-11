from config.config import Config
from learner.train import Trainer
from dataset.manager import TUData

def run():
    config = Config()
    dataset = TUData(config, name="MUTAG")
    print(len(dataset))
    trainer = Trainer(config, dataset.input_dim, dataset.output_dim)
    loader = dataset.get_loader('train')
    trainer.fit(loader)
    samples = trainer.model.sample(10)
    print(samples)

if __name__ == "__main__":
    run()



