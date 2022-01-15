#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
import os

logger = logging.getLogger(__name__)
wandb_logger = lambda dir, version: WandbLogger(
    name="wandb", save_dir=dir, version=version
)
csvlogger = lambda dir, version: CSVLogger(dir, name="csvlogs", version=version)
tblogger = lambda dir, version: TensorBoardLogger(dir, name="tblogs", version=version)


get_loggers = lambda dir, version: [
    wandb_logger(dir, version),
    tblogger(dir, version),
    csvlogger(dir, version),
]
# imports all implemented models and the corresponding string to function mapper
from src.models import *
from src.dataset.loaders import get_loaders


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model_mapper[self.hparams.arch](**self.hparams.arch_params)

    def forward(self, x):
        x = self.model(x)

    def training_step(self, batch, batch_idx):
        # print(f"in train step")
        xb, yb = batch
        y_pred = self.model(xb)
        # y_true = y.squeeze().type(torch.LongTensor)
        if self.hparams.loss == "cross_entropy":
            loss = F.cross_entropy(y_pred, yb)
            acc = torch.mean((y_pred.argmax(dim=-1) == yb).type(torch.float32))
            self.log("train_acc", acc, on_step=False, on_epoch=True)
        else:
            raise NotImplementedError

        self.log("train_loss", loss)
        # import pdb;pdb.set_trace()
        # print(f'batch indx:{batch_idx},{loss.item()}')
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        y_pred = self.model(xb)
        # y_hat = torch.argmax(y_hat, dim=1)
        # import pdb;pdb.set_trace()
        if self.hparams.loss == "cross_entropy":
            acc = torch.mean((y_pred.argmax(dim=-1) == yb).type(torch.float32))
        else:
            raise NotImplementedError
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return acc

    def testing_step(self, batch, batch_idx):
        print(f"in test step")
        pass

    def configure_optimizers(self):
        optim_params = self.hparams.optimizer.optim_params
        if self.hparams.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **optim_params)
        return optimizer


@hydra.main(config_path="config", config_name="default")
def run(cfg):
    # The decorator is enough to let Hydra load the configuration file.
    # Simple logging of the configuration
    logger.info(OmegaConf.to_yaml(cfg))

    # We recover the original path of the dataset:
    path = Path(cfg.data.path)

    # Load the data
    bs = cfg.data.batch_size
    data_type = cfg.data.dataset
    frac = cfg.data.split
    if cfg.is_training:
        # load data loaders
        train_loader = get_loaders(
            cfg.data.path, "train", bs, data_type=data_type, frac=frac[0]
        )
        val_loader = get_loaders(
            cfg.data.path, "valid", bs, data_type=data_type, frac=frac[0]
        )
        print(f"Got Loaders")
        # seed everything for reproducibility
        pl.seed_everything(cfg.seed)
        dir = cfg.artifacts_loc
        # logging
        # to set the version name,here temp is chosen as it might be used in sweep thus to create seperate folders
        version = str(cfg.version) + "_" + str(cfg.model.arch_params.temp)
        logger_list = get_loggers(dir, version)
        # call -backs
        cbs = []
        if "early_stop" in cfg.model.cbs:
            params = cfg.model.cbs.early_stop
            earlystopcb = EarlyStopping(**params, min_delta=0.00, verbose=False)
            cbs.append(earlystopcb)
        if "checkpoint" in cfg.model.cbs:
            store_path = dir + "ckpts/" + str(cfg.version) + "/"
            isExist = os.path.exists(store_path)
            if not isExist:
                os.makedirs(store_path)
            fname = "{epoch}-{val_acc:.2f}"
            params = cfg.model.cbs.checkpoint
            checkptcb = ModelCheckpoint(**params, dirpath=store_path, filename=fname)
            cbs.append(checkptcb)

        # some stupid fuckery - to get sweeps working with hydra.
        # Initialize the W&B agent using the default values from cfg
        config = {
            "temp": cfg.model.arch_params.temp,
        }
        wandb.init(project="unestimates", config=config)
        #'lr': cfg.model.optimizer.optim_params.lr,
        # Get the (possibly updated) values from wandb
        cfg.model.arch_params.temp = wandb.config.temp
        # cfg.model.optimizer.optim_params.lr = wandb.config.lr

        # model
        net = Model(cfg.model)
        trainer = pl.Trainer(
            logger=logger_list, callbacks=cbs, deterministic=True, **cfg.trainer
        )
        trainer.fit(net, train_loader, val_loader)

    else:
        # ?write validation code
        raise NotImplementedError


if __name__ == "__main__":
    run()
