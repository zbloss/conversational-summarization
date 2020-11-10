# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from glob import glob
import json
import torch
from torch.utils.data.dataset import TensorDataset, random_split
from transformers import BartTokenizer
import numpy as np


def text_cleaner(text: str):
    """
    Removes \r and \n from a text string.

    :param text: (str) the text you want cleaned.
    :returns: (str) the cleaned text
    """
    text = text.replace("\r", "")
    text = text.replace("\n", " ")
    return text


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    :param input_filepath: (str) the directory that contains your json files.
    :param output_filepath: (str) the directory you want to store your train/test/val data in.
    :returns: None
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    files = glob(f"{input_filepath}/*.json")

    data = []
    for file in files:
        with open(file, "r") as in_file:
            data += json.load(in_file)
            in_file.close()

    dialogues = [text_cleaner(d["dialogue"]) for d in data]
    summaries = [text_cleaner(d["summary"]) for d in data]
    logger.info("seperated dialogues and summaries")

    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    dialogue_tokens = tokenizer.prepare_seq2seq_batch(
        dialogues, padding="longest", truncation=True, return_tensors="np"
    )

    summary_tokens = tokenizer.prepare_seq2seq_batch(
        summaries, padding="longest", truncation=True, return_tensors="np"
    )

    logger.info("Tokenized Texts")

    num_dialogues, longest_dialogue = dialogue_tokens["input_ids"].shape
    num_summaries, longest_summary = summary_tokens["input_ids"].shape
    assert num_dialogues == num_summaries

    dialogue_lens = np.sum(dialogue_tokens["attention_mask"], axis=1)
    summary_lens = np.sum(summary_tokens["attention_mask"], axis=1)
    ratio = summary_lens / dialogue_lens

    dialogues = np.array(dialogues)[ratio < 1]
    summaries = np.array(summaries)[ratio < 1]

    logger.info(
        "Removed instances where the summary was equal to or longer than the dialogue."
    )

    dialogue_tokens = tokenizer.prepare_seq2seq_batch(
        dialogues.tolist(), padding="longest", truncation=True, return_tensors="pt"
    )

    summary_tokens = tokenizer.prepare_seq2seq_batch(
        summaries.tolist(), padding="longest", truncation=True, return_tensors="pt"
    )
    logger.info("Tokenized into PyTorch Tensors")

    num_dialogues, longest_dialogue = dialogue_tokens["input_ids"].shape
    num_summaries, longest_summary = summary_tokens["input_ids"].shape
    assert num_dialogues == num_summaries

    dataset = TensorDataset(
        dialogue_tokens["input_ids"],
        dialogue_tokens["attention_mask"],
        summary_tokens["input_ids"],
    )

    train_size = int(dialogue_tokens["input_ids"].shape[0] * 0.80)
    test_size = int(dialogue_tokens["input_ids"].shape[0] * 0.10)
    val_size = int(dialogue_tokens["input_ids"].shape[0]) - train_size - test_size

    assert train_size + test_size + val_size == int(
        dialogue_tokens["input_ids"].shape[0]
    )

    train, test, val = random_split(
        dataset=dataset, lengths=(train_size, test_size, val_size)
    )

    torch.save(train, f"{output_filepath}/train_dataset.pt")
    logger.info("Saved train_dataset.pt...")

    torch.save(test, f"{output_filepath}/test_dataset.pt")
    logger.info("Saved test_dataset.pt...")

    torch.save(val, f"{output_filepath}/val_dataset.pt")
    logger.info("Saved val_dataset.pt...")

    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
