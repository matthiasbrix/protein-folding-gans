import time
import argparse

import torch
import torch.utils.data
import torchvision
import numpy as np

from models.gan import Gan, Generator, Discriminator
from models.dcgan import Dcgan, Generator, Discriminator
from model_params import get_model_data_gan, get_model_data_dcgan
from directories import Directories
from dataloader import DataLoader, calc_pairwise_distances
from sampling import gan_sampling, dcgan_sampling
#from openprotein.util import calc_pairwise_distances

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EpochMetrics():
    def __init__(self):
        self.g_loss_acc, self.d_loss_acc = 0.0, 0.0

    def compute_batch_train_metrics(self, g_loss, d_loss):
        self.g_loss_acc += g_loss
        self.d_loss_acc += d_loss

class Training(object):
    def __init__(self, solver):
        self.solver = solver
        self.max_num_fragments = self.solver.data_loader.max_sequence_length()//self.solver.data_loader.residue_fragments

    def _train_batch(self, epoch_metrics, x):
        x = x.to(DEVICE)
        batch_size = x.shape[0]
        valid = torch.ones(batch_size, 1).to(DEVICE) # Discriminator Label to real
        fake = torch.zeros(batch_size, 1).to(DEVICE) # Discriminator Label to fake
        # -----------------
        #  Train Generator
        # -----------------
        self.solver.optimizer_G.zero_grad()
        # Sample noise as generator input, N x 100 x res x res
        z = torch.randn((batch_size, self.solver.model.z_dim, *self.solver.data_loader.img_dims)).to(DEVICE)
        # Generate a batch of images
        gen_imgs = self.solver.generator(z)
        gen_imgs = torch.clamp(gen_imgs, min=0.0001) # clamp values above zero
        gen_imgs = (gen_imgs + gen_imgs.transpose(3, 2))/2 # set symmetric, transpose only spatial dims
        gen_imgs = gen_imgs.expand((gen_imgs.shape[0], self.max_num_fragments, gen_imgs.shape[2], gen_imgs.shape[3]))
        dgz = self.solver.discriminator(gen_imgs)
        valid = torch.ones_like(dgz).to(DEVICE)
        # Loss measures generator's ability to fool the discriminator
        # minimizing the log of the inverted probability of the discriminator’s prediction of fake images
        g_loss = self.solver.model.loss(dgz, valid)
        g_loss.backward()
        self.solver.optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # D is a binary classifier
        self.solver.optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        # D(x) represents the probability with which D thinks that x belongs to p_{data}
        dx = self.solver.discriminator(x)
        valid = torch.ones_like(dx).to(DEVICE)
        real_loss = self.solver.model.loss(dx, valid)
        dgz = self.solver.discriminator(gen_imgs.detach())
        fake = torch.zeros_like(dgz)
        fake_loss = self.solver.model.loss(dgz, fake)
        # The addition of these values means that lower average values of this loss function
        # result in better performance of the discriminator.
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        self.solver.optimizer_D.step()

        epoch_metrics.compute_batch_train_metrics(g_loss.item(), d_loss.item())

    def train(self, epoch_metrics):
        self.solver.model.train()
        for _, train_batch in enumerate(self.solver.data_loader.train_loader):
            if self.solver.data_loader.with_labels:
                x, _ = train_batch[0], train_batch[1]
                self._train_batch(epoch_metrics, x)
            else:
                _, tertiary_positions, _ = train_batch # (original_aa_string, actual_coords_list, mask) = train_batch
                # length of tertiary_positions equals batch_size
                contact_map = torch.zeros((len(tertiary_positions), self.max_num_fragments,\
                    self.solver.data_loader.residue_fragments,\
                    self.solver.data_loader.residue_fragments))
                for protein_idx in range(len(tertiary_positions)):
                    # tertiary_positions[protein_idx] yields the (variable) protein length
                    protein_length = len(tertiary_positions[protein_idx])
                    num_fragments_extract = protein_length//self.solver.data_loader.residue_fragments
                    # 3x3 is for each residue (amino acid), that is each for N, C_alpha, and C' atom where we have the 3D coordinates
                    residues = np.reshape(tertiary_positions[protein_idx], ((protein_length, 3, 3)))
                    for fragment_id in range(num_fragments_extract):
                        fragment_residue = residues[fragment_id*self.solver.data_loader.residue_fragments:fragment_id+1*self.solver.data_loader.residue_fragments] # extracting fragments
                        # in case we extract remaining that is < residue_fragments size, we pad
                        if fragment_residue.shape[0] < self.solver.data_loader.residue_fragments:
                            result = np.zeros((self.solver.data_loader.residue_fragments, 3, 3))
                            result[:fragment_residue.shape[0]] = fragment_residue
                            fragment_residue = torch.Tensor(result).type(torch.float)
                        # residues is res_frag x 3 x 3
                        # TODO: resA and resB????
                        resA, resB = self._atom_filter(fragment_residue) # usually res_frag x 3 or just how many we have (so if all aminoacid -> res_frag x 9)
                        contact_map[protein_idx, fragment_id] = calc_pairwise_distances(resA, resB, torch.cuda.is_available())
                # scale down the contact map
                if self.solver.data_loader.residue_fragments == 16:
                    contact_map /= 10
                elif self.solver.data_loader.residue_fragments == 64:
                    contact_map /= 100
                elif self.solver.data_loader.residue_fragments == 128:
                    contact_map /= 100
                self._train_batch(epoch_metrics, contact_map)

    def _atom_filter(self, residues):
        if self.solver.data_loader.atom == "nitrogen":
            resA = np.reshape(residues, (len(residues), 9))[:, :3]
        elif self.solver.data_loader.atom == "calpha":
            resA = np.reshape(residues, (len(residues), 9))[:, 3:6]
        elif self.solver.data_loader.atom == "cprime":
            resA = np.reshape(residues, (len(residues), 9))[:, 6:]
        else:
            resA = np.reshape(residues, (len(residues), 9))
        resB = resA
        return resA, resB

class Solver():
    def __init__(self, model, generator, discriminator, epochs, data_loader, optimizer_G, optimizer_D,\
        optim_config_G, optim_config_D, num_samples=100, save_model_state=False):
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
        self.epoch = 0
        self.epochs = epochs
        self.train_loss_history = {x: [] for x in ["epochs", "g_loss", "d_loss"]}
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
        #num_train_samples = self.data_loader.num_train_samples
        num_batch_samples = self.data_loader.num_train_batches
        g_loss = metrics.g_loss_acc/num_batch_samples
        d_loss = metrics.d_loss_acc/num_batch_samples
        self.train_loss_history["epochs"].append(epoch)
        self.train_loss_history["g_loss"].append(g_loss)
        self.train_loss_history["d_loss"].append(d_loss)
        return g_loss, d_loss

    def _sample(self, epoch):
        if not self.data_loader.directories.make_dirs:
            return
        with torch.no_grad():
            num_samples = min(self.num_samples, self.data_loader.batch_size)
            if self.data_loader.dataset == "mnist":
                sample = gan_sampling(self.generator, self.model.z_dim, num_samples)
                torchvision.utils.save_image(sample.view(num_samples, *self.data_loader.img_dims),\
                    self.data_loader.directories.result_dir + "/generated_sample_" + str(epoch)\
                    + "_z=" + str(self.model.z_dim) + ".png", nrow=10, normalize=True)
            elif self.data_loader.dataset == "proteins":
                sample = dcgan_sampling(self.generator, self.model.z_dim, self.data_loader.img_dims, num_samples)
                torchvision.utils.save_image(sample.view(num_samples, *sample.shape[1:]),\
                    self.data_loader.directories.result_dir + "/generated_sample_" + str(epoch)\
                    + "_z=" + str(self.model.z_dim) + ".png", nrow=6)

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
        """The main method which training/testing of a model begins"""
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
            g_loss, d_loss = self._save_train_metrics(epoch, epoch_metrics)
            print("====> Epoch: {} g_loss / d_loss avg: {:.4f} / {:.4f}".format(epoch, g_loss, d_loss))
            self._sample(epoch)
            if self.save_model_state:
                self.epoch = epoch+1 # signifying to continue from epoch+1 on.
                torch.save(self, self.data_loader.directories.result_dir + "/model_state.pt")
            print("{:.2f} seconds for epoch {}".format(time.time() - epoch_watch, epoch))
        self._save_final_model()
        print("+++++ RUN IS FINISHED +++++")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training a model (GAN)")
    parser.add_argument("--model", help="Set model to GAN/DCGAN (required)", required=True)
    parser.add_argument("--dataset", help="Set dataset to MNIST/PROTEINS accordingly (required, not case sensitive)",\
        required=True)
    parser.add_argument("--training_file", help="Set the training file to use for protein sequences (optional)",\
        required=False)
    parser.add_argument("--validation_file", help="Set the validation file to use for protein sequences (optional)",\
        required=False)
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

    if model_arg.lower() == "gan":
        data = get_model_data_gan(dataset_arg)
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg)
        model = Gan(data_loader.input_dim, data["z_dim"])
        generator = Generator(data["z_dim"], data_loader.input_dim, data_loader.img_dims)
        discriminator = Discriminator(data_loader.input_dim, 1)
        solver = Solver(model, generator, discriminator, data["epochs"], data_loader, data["optimizer_G"],
                data["optimizer_D"], data["optim_config_G"], data["optim_config_D"], save_model_state=save_model_state)
    elif model_arg.lower() == "dcgan":
        training_file = args["training_file"]
        validation_file = args["validation_file"]
        residue_fragments= args["residue_fragments"]
        data = get_model_data_dcgan(dataset_arg)
        directories = Directories(model_arg.lower(), dataset_arg, data["z_dim"],\
            make_dirs=save_files)
        data_loader = DataLoader(directories, data["batch_size"], dataset_arg.lower(),
                         training_file=training_file, validation_file=validation_file,
                         residue_fragments=residue_fragments, atom="calpha")
        model = Dcgan(data_loader.input_dim, data["z_dim"])
        generator = Generator(data["z_dim"], data_loader.input_dim, data_loader.img_dims, res=residue_fragments)
        discriminator = Discriminator(data_loader.max_sequence_length()//data_loader.residue_fragments,\
                        1, res=residue_fragments)
        solver = Solver(model, generator, discriminator, data["epochs"], data_loader, data["optimizer_G"],
                data["optimizer_D"], data["optim_config_G"], data["optim_config_D"], save_model_state=save_model_state)
    solver.main()
