import os
import json
import torch
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)
pretrained_nlp_model = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(pretrained_nlp_model)


def model_fn(model_dir):
    logger = logging.getLogger(__name__)
    logger.info("Loading the model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    logger.info("Done loading model")
    return model


def input_fn(request_body, content_type="application/json"):
    logger = logging.getLogger(__name__)
    logger.info("Deserializing the input data.")
    if content_type == "application/json":
        input_data = json.loads(request_body)
        input_text = input_data["text"]
        #logger.info(f"input_data: {input_data}")

        tokens = tokenizer.prepare_seq2seq_batch(
            [input_text],
            padding=tokenizer.max_len,
            truncation=True,
            return_tensors="pt",
        )

        return tokens
    raise Exception(f"Invalid ContentType: {content_type}")


def predict_fn(input_data, model):
    logger = logging.getLogger(__name__)
    logger.info("Making Predictions.")

    if torch.cuda.is_available():
        input_data = input_data.cuda()

    with torch.no_grad():
        model.eval()
        output = model.bart.generate(
            input_data["input_ids"], num_beams=4, max_length=90, early_stopping=True
        )
    return output


def output_fn(prediction_output, accept="application/json"):
    logger = logging.getLogger(__name__)
    logger.info("Serializing Model Output.")
    summary = [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for g in prediction_output
    ][0]
    return json.dumps(summary), accept
