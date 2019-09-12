import os

class Directories():
    def __init__(self, model_name, dataset, z_dim, make_dirs=True):
        self.model_name = model_name
        self.data_dir_prefix = "../data/"
        self.results_dir_prefix = "../results/"
        self.result_dir = model_name + "/" + dataset + "_z=" + str(z_dim) + "_0"
        self._dir_index()
        self.result_dir_no_prefix = self.result_dir
        self.result_dir = self.results_dir_prefix + self.result_dir
        if make_dirs:
            self._prepare_directories()
        self.make_dirs = make_dirs

    # makes all relevant directories
    def _prepare_directories(self):
        os.makedirs(self.results_dir_prefix, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    # checks if folder already exists and return directory index
    def _dir_index(self):
        if not os.path.isdir(self.results_dir_prefix+self.result_dir):
            return
        expand = 0
        tmp = self.result_dir.split("_")
        new_dir_name = self.result_dir
        while os.path.isdir(self.results_dir_prefix+new_dir_name):
            expand += 1
            tmp[-1] = str(expand)
            new_dir_name = "_".join(tmp)
        self.result_dir = new_dir_name
        