import os
import torch
import shutil
import argparse
from t5_lightning_module import T5LightningModule

import pytorch_lightning as pl


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CLI for model training")
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        default=4,
        help="Number of data samples to include per forward pass.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=False,
        default=3e-05,
        help="Initial Learning Rate to use.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        default="t5-small",
        help="Hugginface Model name or path to model files.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=True,
        default="dummy-job",
        help="The sagemaker job name, used to name the TensorBoard logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        default=1,
        help="Number of passes through the training data.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        required=False,
        default=1,
        help="Number of gpus to use during training.",
    )
    parser.add_argument(
        "-o",
        "--output-data-dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"],
        help="Directory to store output results",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Directory to store model artifacts",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=os.path.join(os.environ["SM_CHANNEL_TRAIN"], "t5_train_dataset.pt"),
        help="Path to train dataset file",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=os.path.join(os.environ["SM_CHANNEL_TEST"], "t5_test_dataset.pt"),
        help="Path to test dataset file",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default=os.path.join(os.environ["SM_CHANNEL_VAL"], "t5_val_dataset.pt"),
        help="Path to validation dataset file",
    )

    args, _ = parser.parse_known_args()
    print(args)
    
    # Creating our main lightning module
    model = T5LightningModule(
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        val_dataset=args.val_dataset,
        pretrained_nlp_model=args.model_path,
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
    )

    from pytorch_lightning.loggers import TensorBoardLogger
    # Creating the Tensorboard logging module
    tb_logger = TensorBoardLogger(save_dir="/opt/tb_logs", name=args.job_name)

    
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    # Creating the Early Stopping module
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=0.01, patience=3, verbose=False, mode="min"
    )

    
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_logger = LearningRateMonitor(logging_interval='step')
    
    
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    model_checkpoint = ModelCheckpoint(
        filepath=args.model_dir, 
        monitor="val_loss", 
        mode="min", 
        save_top_k=1
    )

    trainer_params = {
        "max_epochs": int(args.epochs),
        "default_root_dir": args.output_data_dir,
        "gpus": int(args.gpus),
        "logger": tb_logger,
        "early_stop_callback": early_stop,
        "checkpoint_callback": model_checkpoint,
        "callbacks": [lr_logger]
    }

    if trainer_params["gpus"] > 1:
        trainer_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model)
    
    results = trainer.test()
    print(f'RESULTS: {results}')

    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.t5.state_dict(), f)
        
    print('Final Model State Dict Saved')