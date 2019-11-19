import time
import argparse

import torch
import torch.utils.data
import torchvision

from models.gan import Gan, Generator, Discriminator
from models.dcgan import Dcgan, Generator, Discriminator
from model_params import get_model_data_gan, get_model_data_dcgan
from directories import Directories
from dataloader import DataLoader
from sampling import gan_sampling, dcgan_sampling
from plots import contact_map_grid

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EpochMetrics():
    def __init__(self):
        self.g_loss_acc, self.d_loss_real_acc = 0.0, 0.0
        self.d_loss_fake_acc, self.d_loss_acc = 0.0, 0.0

    def compute_batch_train_metrics(self, g_loss, d_loss_real, d_loss_fake, d_loss):
        self.g_loss_acc += g_loss
        self.d_loss_real_acc += d_loss_real
        self.d_loss_fake_acc += d_loss_fake
        self.d_loss_acc += d_loss

class Testing(object):
    def __init__(self, generator, z_dim, optimizer_G, test_loader, path,\
        steps=3000, step_size=10, gamma=0.97):
        self.generator = generator
        self.z_dim = z_dim
        self.test_loss = torch.nn.MSELoss()
        self.optimizer_G = optimizer_G
        self.test_loader = test_loader
        self.mse_loss = 0.0
        self.kl_loss = 0.0
        self.path = path
        self.steps = steps
        self.steps_taken = 0
        self.step_size = step_size
        self.gamma = gamma

    def kl_divergence_z(self, z):
        mean = torch.mean(z)
        variance = torch.var(z)
        kl_divergence = 1/2 * (variance + mean.pow(2) - 1 - variance.log())
        return kl_divergence

    # testing the "complexity" of the GAN as defined by Anand and Huang, supplementary material
    # find a z such that G(z) \in x
    def _test_batch(self, x):
        batch_size = x.shape[0]
        z = torch.randn((batch_size, self.z_dim, 1, 1)).to(DEVICE)
        gz = self.generator(z)
        # no clamping/symmetric operations as that is done only during training!
        # ||G(z) - x||_2 + \gamma D_{KL}[N(\mu(z), \sigma^2(z))||N(0,1)]
        mse = self.test_loss(gz, x)
        kl = self.kl_divergence_z(gz)
        loss = mse + 10*kl
        loss.backward()
        self.optimizer_G.step()
        self.mse_loss += mse.item()
        self.kl_loss += kl.item()
        self.steps_taken += 1

    def test(self):
        print("Testing complexity of the GAN, steps: {}".format(self.steps))
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.step_size, gamma=self.gamma)
        self.generator.train()
        epoch = 1
        run_while = True
        while(run_while):
            self.mse_loss = 0.0
            self.kl_loss = 0.0
            for _, test_batch in enumerate(self.test_loader):
                contact_map = test_batch
                self._test_batch(contact_map)
                if self.steps_taken >= self.steps:
                    run_while = False
                scheduler.step()
            print("====> Epoch/Steps: {}/{} mse {:.4f} kl {:.4f}".format(
                epoch, self.steps_taken, self.mse_loss/len(self.test_loader),\
                self.kl_loss/len(self.test_loader)))
            # save the model here
            torch.save(self.generator.state_dict(), self.path)
            epoch += 1

class Training(object):
    def __init__(self, solver):
        self.solver = solver

    def _train_generator(self, gz):
        loss = 0.0
        for i in range(self.solver.g_updates):
            self.solver.optimizer_G.zero_grad()
            dgz = self.solver.discriminator(gz)
            real = torch.ones_like(dgz).to(DEVICE)
            # Loss measures generator's ability to fool the discriminator
            # minimizing the log of the inverted probability of the discriminator’s prediction of fake images
            g_loss = self.solver.model.loss(dgz, real)
            if i == (self.solver.g_updates-1):
                g_loss.backward()
            else:
                g_loss.backward(retain_graph=True)
            loss += g_loss.item()
            self.solver.optimizer_G.step()
        return loss/self.solver.g_updates

    def _train_discriminator(self, x, gz):
        # D is a binary classifier
        self.solver.optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        # D(x) represents the probability with which D thinks that x belongs to p_{data}
        x.unsqueeze_(1)
        # real
        dx = self.solver.discriminator(x)
        real = torch.FloatTensor([self.solver.one_sided_labeling]).repeat(x.shape[0]).to(DEVICE) # ONE SIDED LABELING
        real_loss = self.solver.model.loss(dx.reshape(-1).to(DEVICE), real)
        real_loss.backward()
        # fake
        dgz = self.solver.discriminator(gz)
        fake = torch.zeros_like(dgz).to(DEVICE)
        fake_loss = self.solver.model.loss(dgz, fake)
        fake_loss.backward()
        # The addition of these values means that lower average values of this loss function
        # result in better performance of the discriminator.
        # Add the gradients from the all-real and all-fake batches
        d_loss = real_loss + fake_loss
        self.solver.optimizer_D.step()
        return real_loss.item(), fake_loss.item(), d_loss.item()

    def _train_batch(self, epoch_metrics, x):
        x = x.to(DEVICE)
        batch_size = x.shape[0]
        # Sample noise as generator input, N x 100 x res x res
        z = torch.randn((batch_size, self.solver.model.z_dim, 1, 1)).to(DEVICE)
        # Generate a batch of images
        gz = self.solver.generator(z)
        gz = torch.clamp(gz, min=0.001) # clamp values above zero to ensure positive values
        # settings symmetric here because contact map is only distance to subsequent residues
        gz = (gz + gz.transpose(3, 2))/2 # set symmetric, transpose only spatial dims
        d_loss_real, d_loss_fake, d_loss = self._train_discriminator(x, gz.detach())
        g_loss = self._train_generator(gz)
        epoch_metrics.compute_batch_train_metrics(g_loss, d_loss_real, d_loss_fake, d_loss)

    def train(self, epoch_metrics):
        self.solver.generator.train()
        self.solver.discriminator.train()
        for _, train_batch in enumerate(self.solver.data_loader.train_loader):
            if self.solver.data_loader.dataset == "mnist":
                x, _ = train_batch[0], train_batch[1]
                self._train_batch(epoch_metrics, x)
            elif self.solver.data_loader.dataset == "proteins":
                contact_map = train_batch
                # scale down the contact map
                if self.solver.data_loader.residue_fragments == 16:
                    contact_map /= 10
                elif self.solver.data_loader.residue_fragments == 64:
                    contact_map /= 100
                elif self.solver.data_loader.residue_fragments == 128:
                    contact_map /= 100
                else:
                    raise ValueError("Scaling down went wrong!")
                self._train_batch(epoch_metrics, contact_map)

class Solver():
    def __init__(self, model, generator, discriminator, epochs, data_loader, optimizer_G, optimizer_D,\
        optim_config_G, optim_config_D, max_sequence_length, one_sided_labeling, g_updates,\
        num_samples=100, save_model_state=False):
        self.data_loader = data_loader
        self.model = model
        self.generator = generator
        self.generator.to(DEVICE)
        self.discriminator = discriminator
        self.discriminator.to(DEVICE)
        self._set_weight_decay(optim_config_G)
        self._set_weight_decay(optim_config_D)
        self.optimizer_G = optimizer_G(self.generator.parameters(), **optim_config_G)
        self.optimizer_D = optimizer_D(self.discriminator.parameters(), **optim_config_D)
        self.max_sequence_length = max_sequence_length
        self.epoch = 0
        self.epochs = epochs
        self.train_loss_history = {x: [] for x in ["epochs", "g_loss", "d_loss_real", "d_loss_fake", "d_loss"]}
        self.num_samples = num_samples
        self.one_sided_labeling = one_sided_labeling
        self.g_updates = g_updates

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
        num_batch_samples = self.data_loader.num_train_batches
        g_loss = metrics.g_loss_acc/num_batch_samples
        d_loss_real = metrics.d_loss_real_acc/num_batch_samples
        d_loss_fake = metrics.d_loss_fake_acc/num_batch_samples
        d_loss = metrics.d_loss_acc/num_batch_samples
        self.train_loss_history["epochs"].append(epoch)
        self.train_loss_history["g_loss"].append(g_loss)
        self.train_loss_history["d_loss_real"].append(d_loss_real)
        self.train_loss_history["d_loss_fake"].append(d_loss_fake)
        self.train_loss_history["d_loss"].append(d_loss)
        return g_loss, d_loss_real, d_loss_fake, d_loss

    def get_sample_stats(self):
        if self.data_loader.batch_size >= 32:
            imgs = 25
            rows = 5
            cols = 5
        elif self.data_loader.batch_size == 16:
            imgs = 16
            rows = 4
            cols = 4
        elif self.data_loader.batch_size == 8:
            imgs = 8
            rows = 2
            cols = 4
        elif self.data_loader.batch_size == 4:
            imgs = 4
            rows = 2
            cols = 2
        return imgs, rows, cols

    def _sample(self, epoch):
        if not self.data_loader.directories.make_dirs:
            return
        with torch.no_grad():
            num_samples = min(self.num_samples, self.data_loader.batch_size)
            if self.data_loader.dataset == "mnist":
                sample = gan_sampling(self.generator, self.model.z_dim, num_samples).cpu()
                torchvision.utils.save_image(sample.view(num_samples, *self.data_loader.img_dims),\
                    self.data_loader.directories.result_dir + "/generated_sample_" + str(epoch)\
                    + "_z=" + str(self.model.z_dim) + ".png", nrow=10, normalize=True)
            elif self.data_loader.dataset == "proteins":
                imgs, rows, cols = self.get_sample_stats()
                sample = dcgan_sampling(self.generator, self.model.z_dim, num_samples).cpu()
                contact_map_grid(sample[:imgs], rows=rows, cols=cols, fill=True,\
                    file_name=self.data_loader.directories.result_dir\
                    + "/generated_sample_" + str(epoch)\
                    + "_z=" + str(self.model.z_dim) + ".png")

    # save the model parameters to a txt file in the output folder
    def _save_model_params_to_file(self):
        if not self.data_loader.directories.make_dirs:
            return
        with open(self.data_loader.directories.result_dir + "/model_params_" +\
            self.data_loader.dataset + "_z=" + str(self.model.z_dim) + ".txt", 'w') as param_file:
            params = "epochs: {}\n"\
                "dim(z): {}\n"\
                "batch_size: {}\n"\
                "optimizer_G: {}\n"\
                "optimizer_D: {}\n"\
                .format(self.epochs, self.model.z_dim,\
                    self.data_loader.batch_size, self.optimizer_G,\
                    self.optimizer_D
                )
            params += "dataset: {}\n".format(self.data_loader.dataset)
            params += "img dims: {}\n".format(self.data_loader.img_dims)
            if self.data_loader.dataset == "proteins":
                params += "atom: {}\n".format(self.data_loader.atom)
            params += "Max sequence length: {}\n".format(self.max_sequence_length)
            params += "training file: {}\n".format(self.data_loader.training_file)
            params += "padding: {}\n".format(self.data_loader.padding)
            params += "g_updates: {}\n".format(self.g_updates)
            params += "one_sided_labeling: {}\n".format(self.one_sided_labeling)
            params += str(self.model)
            params += str(self.generator)
            params += str(self.discriminator)
            param_file.write(params)
            print("params used:\n", params)

    # save the final model when training is done.
    def _save_final_model(self):
        if not self.data_loader.directories.make_dirs:
            return
        name = self.data_loader.directories.result_dir + "/model_"
        name += "GAN_"
        name += self.data_loader.dataset + "_z=" + str(self.model.z_dim) + ".pt"
        torch.save(self, name)

    def main(self):
        """The main method where training of a model begins"""
        if self.data_loader.directories.make_dirs:
            print("+++++ START RUN | saved files in {} +++++".format(\
                self.data_loader.directories.result_dir_no_prefix))
        else:
            print("+++++ START RUN +++++ | no save mode")
        self._save_model_params_to_file()
        training = Training(self)
        start = self.epoch if self.epoch else 1
        for epoch in range(start, self.epochs+1):
            epoch_watch = time.time()
            epoch_metrics = EpochMetrics()
            training.train(epoch_metrics)
            g_loss, d_loss_real, d_loss_fake, d_loss = self._save_train_metrics(epoch, epoch_metrics)
            print("====> Epoch: {} g_loss {:.4f} / d_loss_real {:.4f} / d_loss_fake {:.4f} / d_loss avg: {:.4f}".format(
                epoch, g_loss, d_loss_real, d_loss_fake, d_loss))
            self._sample(epoch)
            if self.save_model_state and (epoch % 5) == 0:
                self.epoch = epoch+1 # signifying to continue from epoch+1 on.
                torch.save(self, self.data_loader.directories.result_dir + "/model_state_" + str(epoch) + ".pt")
            print("{:.2f} seconds for epoch {}".format(time.time() - epoch_watch, epoch))
        self._save_final_model()
        print("+++++ RUN IS FINISHED +++++")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training a model (GAN)")
    parser.add_argument("--model", help="Set model to GAN/DCGAN (required)", required=True)
    parser.add_argument("--dataset", help="Set dataset to MNIST/PROTEINS accordingly (required, not case sensitive)",\
        required=True)
    parser.add_argument("--training_file", help="Set the training file to use for protein sequences",\
        required=True)
    parser.add_argument("--max_sequence_length", help="Set the max sequence length", required=True, type=int)
    parser.add_argument("--residue_fragments", help="Set the number of residue fragments (optional)",\
        required=False, type=int)
    parser.add_argument("--save_files", help="Determine if files (samples etc.) should be saved (optional, default: False)",\
        required=False, action='store_true')
    parser.add_argument("--save_model_state", help="Determine if state of model should be saved after each epoch\
        during training (optional, default: False)", required=False, action='store_true')

    args = vars(parser.parse_args())
    model_arg = args["model"]
    dataset_arg = args["dataset"]
    save_files = args["save_files"]
    save_model_state = args["save_model_state"]
    max_sequence_length = args["max_sequence_length"]

    if model_arg.lower() == "gan":
        data = get_model_data_gan(dataset_arg)
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg)
        model = Gan(data_loader.input_dim, data["z_dim"])
        generator = Generator(data["z_dim"], data_loader.input_dim, data_loader.img_dims)
        discriminator = Discriminator(data_loader.input_dim, 1)
    elif model_arg.lower() == "dcgan":
        training_file = args["training_file"]
        residue_fragments= args["residue_fragments"]
        data = get_model_data_dcgan(dataset_arg)
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg.lower(),
                        training_file=training_file, residue_fragments=residue_fragments,\
                        atom="calpha", padding=data["padding"])
        model = Dcgan(data_loader.input_dim, data["z_dim"])
        generator = Generator(data["z_dim"], res=residue_fragments)
        discriminator = Discriminator(1, 1, res=residue_fragments)
    solver = Solver(model, generator, discriminator, data["epochs"], data_loader, data["optimizer_G"],
                    data["optimizer_D"], data["optim_config_G"], data["optim_config_D"], max_sequence_length,\
                    data["one_sided_labeling"], data["g_updates"], save_model_state=save_model_state)
    solver.main()
