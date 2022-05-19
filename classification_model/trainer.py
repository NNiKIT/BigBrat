import copy
import os
import shutil
from typing import Any

import numpy as np
import torch
import wandb
from invoke import Config
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification_model.dataset import CustomDataset
from classification_model.plots import plot_confusion_matrix, plot_loss_and_score


class Trainer:
    def __init__(self, config: Config, model: Any, optimizer: torch.optim, criterion: Any):
        super().__init__()

        self.config = config
        self.best_epoch = 1
        self.best_loss = float("inf")

        self.device = torch.device(self.config.DEVICE if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        if config.weights:
            state = torch.load(self.config.weights)  # load the best weights
            self.model.load_state_dict(state)
            self.best_model = self.model

        # use predefined function to load the image data into workspace
        train_dataset = CustomDataset(config.TRAIN_DIR, resize_shape=config.IMG_SIZE)
        val_dataset = CustomDataset(config.VAL_DIR, resize_shape=config.IMG_SIZE)
        test_dataset = CustomDataset(config.TEST_DIR, resize_shape=config.IMG_SIZE)

        itmo_dataset = CustomDataset(config.Itmo_DIR, resize_shape=config.IMG_SIZE)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True
        )
        self.itmo_loader = DataLoader(
            itmo_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True
        )

        self.resize_shape = tuple(self.config.IMG_SIZE)
        if optimizer:
            self.optimizer = optimizer
        if criterion:
            self.criterion = criterion
        if self.config.wandb.logging:
            wandb.init(
                project=self.config.wandb.project_name,
                name=self.config.wandb.run_name,
                entity=self.config.wandb.team_name,
                config={
                    "epochs": self.config.NUM_EPOCHS,
                    "batch_size": self.config.BATCH_SIZE,
                    "lr": self.config.learning_rate,
                },
            )
        self.log_path = "experiments/exp{}/".format(
            str(len([i for i in os.listdir("experiments/") if i.startswith("exp")]) + 1)
        )
        os.makedirs(self.log_path)

    def make_dirs(self) -> None:
        dirs = ["pics", "weights"]
        for directory in dirs:
            if not os.path.exists(self.log_path + directory):
                os.makedirs(self.log_path + directory)
        print("All files are in " + self.log_path)

    def fit_epoch(self):
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0

        for inputs, labels in tqdm(self.train_loader):
            inputs = inputs.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_data += inputs.size(0)

        train_loss = running_loss / processed_data
        train_acc = running_corrects.cpu().numpy() / processed_data
        return train_loss, train_acc

    def validate(self, val_loader: DataLoader):
        self.model = self.model.to(self.device)
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0

        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_size += inputs.size(0)
        val_loss = running_loss / processed_size
        val_acc = running_corrects.double() / processed_size
        return val_loss, val_acc

    def train(self, validation: str = "edinburgh"):
        history = {"train_loss": [], "val_loss": [], "train_score": [], "val_score": []}

        if validation == "edinburgh":
            val_loader = self.val_loader
        else:
            val_loader = self.itmo_loader

        self.make_dirs()
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"EPOCH #{epoch}")
            print("Training started...")
            train_loss, train_acc = self.fit_epoch()

            print("Validation started...")
            val_loss, val_acc = self.validate(val_loader)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model = copy.deepcopy(self.model)
                pred_best_epoch = self.best_epoch
                self.best_epoch = epoch
                torch.save(
                    self.best_model.state_dict(),
                    self.log_path + f"weights/best_model_{self.best_epoch+1}ep.pt",
                )
                print("SAVED: ", self.log_path + f"weights/best_model_{self.best_epoch+1}ep.pt")
                if os.path.exists(self.log_path + f"weights/best_model_{pred_best_epoch+1}ep.pt"):

                    os.remove(self.log_path + f"weights/best_model_{pred_best_epoch+1}ep.pt")

                    print(
                        "DELETED: ",
                        self.log_path + f"weights/best_model_{pred_best_epoch+1}ep.pt",
                    )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_score"].append(train_acc)
            history["val_score"].append(val_acc.to("cpu").numpy())

            torch.save(
                self.model.state_dict(), self.log_path + f"weights/last_model_{epoch+1}ep.pt"
            )
            print("\nSAVED: ", self.log_path + f"weights/last_model_{epoch+1}ep.pt")

            if os.path.exists(self.log_path + f"weights/last_model_{epoch}ep.pt"):
                os.remove(self.log_path + f"weights/last_model_{epoch}ep.pt")
                print("DELETED: ", self.log_path + f"weights/last_model_{epoch}ep.pt")

            if self.config.wandb.logging:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "train/score": train_acc,
                        "val/score": val_acc,
                    }
                )
            print()

        print("-----------------TRAINING OVER-----------------")
        plot_loss_and_score(
            history["train_loss"],
            history["val_loss"],
            history["train_score"],
            history["val_score"],
            save_path=self.log_path + "pics/loss_and_score.png",
        )
        if self.config.save_torchscript:
            try:
                self.best_model.set_swish(memory_efficient=False)
                model_scripted = torch.jit.trace(
                    self.best_model, torch.Tensor(1, 3, 224, 224).to(self.device)
                )
                model_scripted.save(self.log_path + "weights/best_model_scripted.pt")
            except:
                try:
                    model_scripted = torch.jit.script(self.best_model)
                    model_scripted.save(self.log_path + "weights/best_model_scripted.pt")
                except:
                    print("Can't convert model to torchscript format.")

        if self.config.wandb.logging:
            print(
                "LOGGING:",
                self.log_path + "weights/best_model_{}ep.pt".format(self.best_epoch + 1),
            )
            wandb.log_artifact(
                self.log_path + "weights/best_model_{}ep.pt".format(self.best_epoch + 1),
                name=self.config.wandb.model_name,
                type="model",
            )

            if self.config.save_torchscript:
                if os.path.exists(self.log_path + "weights/best_model_scripted.pt"):
                    wandb.log_artifact(
                        self.log_path + "weights/best_model_scripted.pt",
                        name=self.config.wandb.model_name,
                        type="model",
                    )
                    print("LOGGING:", self.log_path + "weights/best_model_scripted.pt")

            wandb.log({"plots": wandb.Image(self.log_path + "pics/loss_and_score.png")})

        get_conf_mtx = self.config.confusion_matrix
        if get_conf_mtx:
            confusion_mtx = self.get_confusion_matrix(get_conf_mtx)
            path_to_save = self.log_path + "pics/confusion_matrix_{}.png".format(get_conf_mtx)
            plot_confusion_matrix(
                confusion_mtx,
                classes=[0, 1],
                title="Confusion matrix:" + get_conf_mtx,
                normalize=False,
                save_path=path_to_save,
            )
            if self.config.wandb.logging:
                wandb.log({"confusion_matrix": wandb.Image(path_to_save)})
        if self.config.get_metrics:
            y_true, probs = self.predict("edinburgh")
            y_pred = np.argmax(probs, axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            print("Accuracy: ", acc)
            print("F1-score: ", f1)
            print("Recall-score: ", recall)
            if self.config.wandb.logging:
                wandb.summary["accuracy"] = acc
                wandb.summary["F1_score"] = f1
                wandb.summary["Recall_score"] = recall

        if self.config.wandb.logging:
            wandb.finish()
        return history

    def predict(self, mode: str = "edinburgh"):
        if mode == "edinburgh":
            data_loader = self.test_loader
        else:
            data_loader = self.itmo_loader
        y_pred = []
        y_true = []
        self.best_model.eval()

        # iterate over test data
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(self.device)
                output = self.best_model(inputs).cpu()  # Feed Network

                output = torch.nn.functional.softmax(output, dim=-1).numpy()
                y_pred.extend(output)  # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)  # Save Truth
        return y_true, y_pred

    def inference(self, image_path: str):
        image = imread(image_path)
        image = resize(image, self.resize_shape, mode="constant", anti_aliasing=True)

        X = np.array(image, np.float32)
        X = np.rollaxis(X, 2, 0)
        inputs = np.expand_dims(X, axis=0)
        inputs = torch.Tensor(inputs)

        with torch.no_grad():
            inputs = inputs.to(self.device)
            self.best_model.eval()
            logit = self.best_model(inputs).cpu()
            probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        y_pred = np.argmax(probs)
        return y_pred

    def get_confusion_matrix(self, mode: str = "edinburgh"):
        # validate on val set
        print("Dataloader:", mode)
        if mode == "edinburgh":
            data_loader = self.test_loader
        else:
            data_loader = self.itmo_loader
        y_pred = []
        y_true = []
        self.best_model.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader):
                inputs = inputs.to(self.device)
                output = self.best_model(inputs).cpu()

                output = torch.nn.functional.softmax(output, dim=-1).numpy()
                y_pred.extend(output)

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)
        y_pred = np.argmax(y_pred, axis=1)
        confusion_mtx = confusion_matrix(y_true, y_pred)
        return confusion_mtx
