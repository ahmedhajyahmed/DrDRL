from openpyxl import Workbook
import os


class Tracker:
    def __init__(self, dir_path="tracking", env_params_name=None, to_train=True):
        self.dir_path = dir_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if to_train:
            self.training_workbook = Workbook()
            self.training_sheet = self.training_workbook.active
            self.training_sheet.append(["seed", "time", "final_episode", "average_reward"])
        else:
            self.repair_workbook = Workbook()
            self.repair_sheet = self.repair_workbook.active

            if env_params_name is None:
                env_params_name = []

            initial_row = ["instance"] + env_params_name + ["cl_time", "DrDRL_time", "cl_final_episode",
                                                            "DrDRL_final_episode", "cl_total_steps",
                                                            "DrDRL_total_steps", "cl_average_reward",
                                                            "DrDRL_average_reward", "drift_type", "seed"]
            self.repair_sheet.append(initial_row)
        self.continual_learning_data = []
        self.pruning_retraining_data = []

    def save(self):
        self.training_workbook.save(filename=os.path.join(self.dir_path, "training.xlsx"))
        self.repair_workbook.save(filename=os.path.join(self.dir_path, "repair.xlsx"))

    def save_training(self):
        self.training_workbook.save(filename=os.path.join(self.dir_path, "training.xlsx"))

    def save_repair(self):
        self.repair_workbook.save(filename=os.path.join(self.dir_path, "repair.xlsx"))

    def add_training_row(self, row):
        self.training_sheet.append(row)

    def add_repair_row(self, row):
        self.repair_sheet.append(row)

    def add_continual_learning_data(self, data):
        self.continual_learning_data = data

    def add_pruning_retraining_data(self, data):
        self.pruning_retraining_data = data
