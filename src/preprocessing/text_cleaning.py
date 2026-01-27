import re

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    if batch["darija_Arab_new"] is None:
        batch["text"] = "خاوي"
    else:
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["darija_Arab_new"]).lower() + " "
    return batch

def remove_digits(batch):
    if batch["text"] is None:
        batch["text"] = "خاوي"
    else:
        batch["text"] = re.sub(r'\d+', '', batch["text"])
    return batch

def normalize(batch):
    if batch["text"] is None:
        batch["text"] = "خاوي"
    else:
        hamza_pattern = re.compile(r'[ءأإؤئيآ]')
        batch["text"] = re.sub(hamza_pattern, 'ء', batch["text"])
    return batch
