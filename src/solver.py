import time
import argparse
import os

import torch
import torch.utils.data
import torchvision.utils

from models.gan import Gan
from model_params import get_model_data_gan
from directories import Directories
from dataloader import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EpochMetrics():
    def __init__(self):
        self.train_loss_acc, self.test_loss_acc = 0.0, 0.0

    def compute_batch_train_metrics(self, train_loss):
        self.train_loss_acc += train_loss

    def compute_batch_test_metrics(self, test_loss):
        self.test_loss_acc += test_loss

class Training(object):
    def __init__(self, solver):
        self.solver = solver

    def _train_batch(self, epoch_metrics, x):
        self.solver.optimizer.zero_grad()
        x = x.view(-1, self.solver.data_loader.input_dim).to(DEVICE)
        decoded = self.solver.model(x) # TODO decoded?
        loss = self.solver.model.loss_function(decoded, x)
        loss.backward()
        self.solver.optimizer.step()
        epoch_metrics.compute_batch_train_metrics(loss.item())

    def train(self, epoch_metrics):
        self.solver.model.train()
        for _, train_batch in enumerate(self.solver.data_loader.train_loader):
            if self.solver.data_loader.with_labels:
                x, y = train_batch[0], train_batch[1]
                self._train_batch(epoch_metrics, x, y)
            else:
                x = train_batch
                self._train_batch(epoch_metrics, x)

class Testing(object):
    def __init__(self, solver):
        self.solver = solver

    def _test_batch(self, epoch_metrics, batch_idx, epoch, x):
        x = x.view(-1, self.solver.data_loader.input_dim).to(DEVICE)
        decoded = self.solver.model(x)
        loss = self.solver.model.loss_function(decoded, x)
        epoch_metrics.compute_batch_test_metrics(loss.item())
        if batch_idx == 0 and self.solver.data_loader.directories.make_dirs: # check w/ test set on first batch in test set.
            n = min(x.size(0), 16) # 2 x 8 grid
            comparison = torch.cat([x.view(x.size(0), *self.solver.data_loader.img_dims)[:n],\
                decoded.view(x.size(0), *self.solver.data_loader.img_dims)[:n]])
            torchvision.utils.save_image(comparison.cpu(), self.solver.data_loader.directories.result_dir \
                + "/test_reconstruction_" + str(epoch) + "_z=" + str(self.solver.model.z_dim) + ".png", nrow=n)

    def test(self, epoch, epoch_metrics):
        self.solver.model.eval()
        with torch.no_grad():
            for batch_idx, test_batch in enumerate(self.solver.data_loader.test_loader):
                if self.solver.data_loader.with_labels:
                    x, y = test_batch[0], test_batch[1]
                    self._test_batch(epoch_metrics, batch_idx, epoch, x, y)
                else:
                    self._test_batch(epoch_metrics, batch_idx, epoch, test_batch)

class Solver():
    def __init__(self, model, data_loader, optimizer, epochs, optim_config,\
            step_config=None, lr_scheduler=None, num_samples=100, save_model_state=False):
        self.data_loader = data_loader
        self.model = model
        self.model.to(DEVICE)
        self._set_weight_decay(optim_config)
        self.optimizer = optimizer(self.model.parameters(), **optim_config)
        self.epoch = 0
        self.epochs = epochs
        self.step_config = step_config
        self.lr_scheduler = lr_scheduler(self.optimizer, **step_config) if lr_scheduler else lr_scheduler
        self.train_loss_history = {x: [] for x in ["epochs", "train_loss_acc", "recon_loss_acc", "kl_diverg_acc"]}
        self.test_loss_history = []
        self.num_samples = num_samples

        if save_model_state and not self.data_loader.directories.make_dirs:
            raise ValueError("Can't save state if no folder is assigned to this run!")
        self.save_model_state = save_model_state

    def _set_weight_decay(self, optim_config):
        if optim_config["weight_decay"] is None:
            optim_config["weight_decay"] = 0.0
        elif optim_config["weight_decay"] == 1:
            # batch wise regularization, so M/N in all
            optim_config["weight_decay"] = 1/(float(self.data_loader.num_train_samples))

    def _save_train_metrics(self, epoch, metrics):
        num_train_samples = self.data_loader.num_train_samples
        num_train_batches = self.data_loader.num_train_batches
        train_loss = metrics.train_loss_acc/num_train_samples
        recon_loss = metrics.recon_loss_acc/num_train_samples
        kl_div = metrics.kl_diverg_acc/num_train_samples
        self.train_loss_history["epochs"].append(epoch)
        self.train_loss_history["train_loss_acc"].append(train_loss)
        return train_loss

    def _save_test_metrics(self, metrics):
        test_loss = metrics.test_loss_acc/self.data_loader.num_test_samples
        self.test_loss_history.append(test_loss)
        return test_loss

    def _sample(self, epoch):
        if not self.data_loader.directories.make_dirs:
            return
        with torch.no_grad():
            sample = gan_sampling(self.model.decoder, self.num_samples, self.model.z_dim)
            num_samples = min(self.num_samples, sample.size(0))
            torchvision.utils.save_image(sample.view(num_samples, *self.data_loader.img_dims),\
                self.data_loader.directories.result_dir + "/generated_sample_" + str(epoch)\
                + "_z=" + str(self.model.z_dim) + ".png", nrow=10)

    # save the model parameters to a txt file in the output folder
    def _save_model_params_to_file(self):
        if not self.data_loader.directories.make_dirs:
            return
        with open(self.data_loader.directories.result_dir + "/model_params_" +\
            self.data_loader.dataset + "_z=" + str(self.model.z_dim) + ".txt", 'w') as param_file:
            params = "epochs: {}\n"\
                "optimizer: {}\n"\
                "beta: {}\n"\
                "dim(z): {}\n"\
                "batch_size: {}\n"\
                "lr_scheduler: {}\n"\
                "step_config: {}\n"\
                .format(self.epochs, self.optimizer, self.model.beta, self.model.z_dim,\
                    self.data_loader.batch_size, self.lr_scheduler,\
                    self.step_config)
            params += "dataset: {}\n".format(self.data_loader.dataset)
            params += str(self.model)
            param_file.write(params)
            print("params used:", params)

    # save the final model when training is done.
    def _save_final_model(self):
        name = self.data_loader.directories.result_dir + "/model_"
        name += "GAN_"
        last_train_loss = self.train_loss_history["train_loss_acc"][-1]
        name += self.data_loader.dataset + "_train_loss=" + "{0:.2f}".format(last_train_loss)\
            + "_z=" + str(self.model.z_dim) + ".pt"
        torch.save(self, name)

    def main(self):
        """The main method which training/testing of a model begins"""
        if self.data_loader.directories.make_dirs:
            print("+++++ START RUN | saved files in {} +++++".format(\
                self.data_loader.directories.result_dir_no_prefix))
        else:
            print("+++++ START RUN +++++ | no save mode")
        self._save_model_params_to_file()
        training = Training(self)
        testing = Testing(self)
        start = self.epoch if self.epoch else 1
        for epoch in range(start, self.epochs+1):
            epoch_watch = time.time()
            epoch_metrics = EpochMetrics()
            training.train(epoch_metrics)
            train_loss = self._save_train_metrics(epoch, epoch_metrics)
            print("====> Epoch: {} train set loss avg: {:.4f}".format(epoch, train_loss))
            testing.test(epoch, epoch_metrics)
            test_loss = self._save_test_metrics(epoch_metrics)
            print("====> Test set loss avg: {:.4f}".format(test_loss))
            self._sample(epoch)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            if self.save_model_state:
                self.epoch = epoch+1 # signifying to continue from epoch+1 on.
                torch.save(self, self.data_loader.directories.result_dir + "/model_state.pt")
            print("{:.2f} seconds for epoch {}".format(time.time() - epoch_watch, epoch))
        if self.data_loader.directories.make_dirs:
            self._save_final_model()
        print("+++++ RUN IS FINISHED +++++")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training a model (GAN)")
    parser.add_argument("--model", help="Set model to GAN (required)", required=True)
    parser.add_argument("--dataset", help="Set dataset to MNIST/LFW/FF/LungScans accordingly (required, not case sensitive)", required=True)
    parser.add_argument("--save_files", help="Determine if files (samples etc.) should be saved (optional, default: False)", required=False, action='store_true')
    parser.add_argument("--save_model_state", help="Determine if state of model should be saved after each epoch during training (optional, default: False)",\
        required=False, action='store_true')
    parser.add_argument('--scales', help="Enables scaling of the model as specified in model_params", default=None, action='store_true')
    parser.add_argument('--thetas', help="Enables rotations of the model as specified in model_params", default=None, action='store_true')
    args = vars(parser.parse_args())
    model_arg = args["model"]
    dataset_arg = args["dataset"]
    save_files = args["save_files"]
    save_model_state = args["save_model_state"]

    if model_arg.lower() == "gan":
        data = get_model_data_gan(dataset_arg)
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg)
        model = Gan(data_loader.input_dim, data["hidden_dim"],\
            data["z_dim"], data["beta"], data["batch_norm"])
        solver = Solver(model, data_loader, data["optimizer"],\
            data["epochs"], data["optim_config"],\
            step_config=data["step_config"], lr_scheduler=data["lr_scheduler"],\
            save_model_state=save_model_state)
    solver.main()
