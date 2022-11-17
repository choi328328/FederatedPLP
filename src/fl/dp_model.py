from deepctr_torch.models import DCN
from deepctr_torch.layers import CrossNet, DNN
from deepctr_torch.layers.utils import slice_arrays
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from opacus import PrivacyEngine
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
from pyvacy import optim as pyvacy_optim
from .dp_utils import get_epsilon
from loguru import logger

'''
module for differential privacy
'''

class DCN_DP(DCN):
    def __init__(
        self,
        linear_feature_columns,
        dnn_feature_columns,
        cross_num=2,
        cross_parameterization="vector",
        dnn_hidden_units=(128, 128),
        l2_reg_linear=0.00001,
        l2_reg_embedding=0.00001,
        l2_reg_cross=0.00001,
        l2_reg_dnn=0,
        init_std=0.0001,
        seed=1024,
        dnn_dropout=0,
        dnn_activation="relu",
        dnn_use_bn=False,
        task="binary",
        device="cpu",
        gpus=None,
        nm=0.0,
        mgn = 0.0,
        
    ):

        super().__init__(
            linear_feature_columns,
            dnn_feature_columns,
            cross_num=cross_num,
            cross_parameterization=cross_parameterization,
            dnn_hidden_units=dnn_hidden_units,
            l2_reg_linear=l2_reg_linear,
            l2_reg_embedding=l2_reg_embedding,
            l2_reg_cross=l2_reg_cross,
            l2_reg_dnn=l2_reg_dnn,
            init_std=init_std,
            seed=seed,
            dnn_dropout=dnn_dropout,
            dnn_activation=dnn_activation,
            dnn_use_bn=dnn_use_bn,
            task=task,
            device=device,
            gpus=gpus,
        )
        #self.optim =torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss_func = F.binary_cross_entropy
        self.nm = nm
        self.mgn = mgn
     
    def get_target_delta(self, data_size: int) -> float:
        """Generate target delta given the size of a dataset. Delta should be
        less than the inverse of the datasize.
        Parameters
        ----------
        data_size : int
            The size of the dataset.
        Returns
        -------
        float
            The target delta value.
        """
        den = 1
        while data_size // den >= 1:
            den *= 10
        return 1 / den
    
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose=1,
        initial_epoch=0,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        callbacks=None,
    ):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        # pyvacy

        
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                (
                    val_x,
                    val_y,
                    val_sample_weight,
                ) = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    "When passing a `validation_data` argument, "
                    "it must contain either 2 items (x_val, y_val), "
                    "or 3 items (x_val, y_val, val_sample_weights), "
                    "or alternatively it could be a dataset or a "
                    "dataset or a dataset iterator. "
                    "However we received `validation_data=%s`" % validation_data
                )
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0.0 < validation_split < 1.0:
            do_validation = True
            if hasattr(x[0], "shape"):
                split_at = int(x[0].shape[0] * (1.0 - validation_split))
            else:
                split_at = int(len(x[0]) * (1.0 - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at), slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at), slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y)
        )
        if batch_size is None:
            batch_size = 256
        model = self.train()
        
        #compile
        self.metrics={}
        
        if self.gpus:
            print("parallel running on these gpus:", self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)


        
        #### Add Differential privacy(opacus)
        # len_dataset= len(x)
        # alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        # # Get delta
        # delta = self.get_target_delta(len_dataset)
        # # Set the sample rate
        # sample_rate = batch_size / len_dataset
        # # Define the Privacy Engine
        # n_acc_steps = 1# int(vbatch_size / batch_size)
        # privacy_engine = PrivacyEngine()
        # model, optim, train_loader = privacy_engine.make_private(
        # module=model,
        # optimizer=optim,
        # data_loader=train_loader,
        # noise_multiplier=self.nm,
        # max_grad_norm=self.mgn,
        # )
        # privacy_engine.to(self.device)
        # # Attach PrivacyEngine after moving it to the same device as the model
        # privacy_engine.attach(optim) # @note change self.optim to optim
        ###
        
        ### pyvacy
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size
        )
        len_dataset= len(x)
        delta = self.get_target_delta(len_dataset)
        
        self.optim= pyvacy_optim.DPSGD(
        params=self.parameters(),
        l2_norm_clip=self.mgn,
        noise_multiplier=self.nm,
        minibatch_size=batch_size,
        microbatch_size=1,
        lr=0.01,
        weight_decay=0.00001,
        )
        
        eps=get_epsilon(
            len_dataset,
            batch_size,
            self.nm,
            (len_dataset/batch_size)+1,
            delta
        )
        
        ###
        
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, "model"):  # for tf1.4
            callbacks.__setattr__("model", self)
        callbacks.model.stop_training = False

        # Train
        print(
            "Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
                len(train_tensor_data), len(val_y), steps_per_epoch
            )
        )
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            '''          
                            for X_minibatch, y_minibatch in minibatch_loader(train_dataset):
                    optimizer.zero_grad()
                    for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                        X_microbatch = X_microbatch.to(params['device'])
                        y_microbatch = y_microbatch.to(params['device'])

                        optimizer.zero_microbatch_grad()
                        loss = loss_function(classifier(X_microbatch), y_microbatch)
                        loss.backward()
                        optimizer.microbatch_step()
                    optimizer.step()

                    if iteration % 10 == 0:
                        print('[Iteration %d/%d] [Loss: %f]' % (iteration, params['iterations'], loss.item()))
                    iteration += 1
            '''
            
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        
                        self.optim.zero_grad()
                        #y_pred = model(x).squeeze()
                        for X_microbatch, y_microbatch in DataLoader(Data.TensorDataset(x, y), batch_size=1): # TODO : 그냥 x,y로 넣어야하나?
                            X_microbatch = X_microbatch.to('cuda')
                            y_microbatch = y_microbatch.to('cuda')
                            self.optim.zero_microbatch_grad()
                            y_pred = model(X_microbatch)
                            loss = self.loss_func(y_pred.squeeze(), y_microbatch.squeeze())
                            reg_loss = self.get_regularization_loss()
                            total_loss = loss + reg_loss + self.aux_loss
                            total_loss.backward()
                            self.optim.microbatch_step()
                        self.optim.step()
                        # for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
                        #     X_microbatch = X_microbatch.to(params['device'])

                        # optim.zero_grad()
                        # loss = loss_func(y_pred, y.squeeze(), reduction="sum")
                        
                        # reg_loss = self.get_regularization_loss()
                        # total_loss = loss + reg_loss + self.aux_loss

                        # loss_epoch += loss.item()
                        # total_loss_epoch += total_loss.item()
                        # total_loss.backward()
                        # optim.step()


            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print("Epoch {0}/{1}".format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"]
                )

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += (
                            " - "
                            + "val_"
                            + name
                            + ": {0: .4f}".format(epoch_logs["val_" + name])
                        )
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break
        self.epsilon =eps
        logger.info(f"Epsilon: {self.epsilon}, nm :{self.nm},  mgn :{self.mgn}, Epochs:{epochs}" )
        callbacks.on_train_end()

        return self.history
    