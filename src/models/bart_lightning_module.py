import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, BartTokenizer, get_cosine_schedule_with_warmup


class BartLightningModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_nlp_model: str,
        train_dataset: str,
        test_dataset: str,
        val_dataset: str,
        batch_size: int,
        learning_rate: float = 3e-05,
    ):
        """
        A Pytorch-Lightning Module that trains Bart from the  HuggingFace transformers
        library.

        :param pretrained_nlp_model: (str) the name of the pretrained mode you want to use.
        :param train_dataset: (str) path to pytorch dataset containing train data.
        :param test_dataset: (str) path to pytorch dataset containing test data.
        :param val_dataset: (str) path to pytorch dataset containing validation data.
        :param batch_size: (int) Number of data points to pass per batch in the train, test, and validation sets.
        :param learning_rate: (float) Initial Learning Rate to set.
        :returns: None
        """
        super().__init__()

        self.batch_size = int(batch_size)
        self.train_dataset = str(train_dataset)
        self.test_dataset = str(test_dataset)
        self.val_dataset = str(val_dataset)
        self.hparams.learning_rate = learning_rate
        
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_nlp_model)
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_nlp_model)
        
        
    def forward(self, x):
        
        # Run through NLP Model
        output = self.bart(**x)
        return output

    def training_step(self, batch, batch_idx):

        input_ids, attn_mask, labels = batch

        x = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "return_dict": True,
        }

        # Run through NLP Model
        out = self.bart(**x)

        loss = out["loss"]
        print(f"current_epoch: {self.current_epoch};")
        print(f"global_step: {self.global_step};")
        print(f"train_loss: {loss};")
        print(f"learning_rate: {self.hparams.learning_rate};")

        
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch

        x = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "return_dict": True,
        }

        # Run through NLP Model
        out = self.bart(**x)
        loss = out["loss"]
        
        
        print(f"val_loss: {loss};")
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)

        if batch_idx == len(self.val_dataloader())-1:
            predictions = torch.argmax(out['logits'], dim=-1)
            predictions = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            references = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            self.logger.experiment.add_text(
                tag="example_summaries",
                text_string=f"""
                Model Summary: {predictions[0]}
                
                Target Summary: {references[0]}""",
                global_step=self.global_step,
            )

        return loss
        

    def test_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = batch

        x = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "return_dict": True,
        }

        # Run through NLP Model
        out = self.bart(**x)

        loss = out["loss"]
        print(f"test_loss: {loss};")

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True) 

        return loss

    def configure_optimizers(self):
        """
        Recreating the same Adam optimizer used in the author's code.
        """

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
        
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=20000)
        print(f'scheduler: {scheduler}')
        gen_sched = {'scheduler': scheduler, 'interval': 'step'}
        
        return [optimizer], [gen_sched]
    
        
    def train_dataloader(self):
        return DataLoader(
            torch.load(self.train_dataset), shuffle=True, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            torch.load(self.val_dataset), shuffle=False, batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            torch.load(self.test_dataset), shuffle=True, batch_size=self.batch_size
        )
