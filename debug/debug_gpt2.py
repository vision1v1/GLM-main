import os
import torch
from tokenization_gpt2 import GPT2Tokenizer
from modeling_gpt2 import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

torch.set_printoptions(linewidth=500)  # 方便阅读

gpt2_path = os.path.join(os.getenv('my_data_dir'), "pretrained", "gpt2")


def debug_bpe():
    """
    调试 gpt2 的bpe
    """
    tokenizer:GPT2Tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
    token = 'Tsinghua'
    output = tokenizer.bpe(token)
    print(output)


def debug_tokenize():
    """
    分析 gpt2 的分词
    """
    text = "Tsinghua University is located in"
    tokenizer:GPT2Tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
    output = tokenizer.tokenize(text=text)
    print(output)


def debug_tokenizer():
    # text = "Replace me by any text you'd like."
    text = "Tsinghua University is located in [MASK]."
    tokenizer:GPT2Tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
    # encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = tokenizer(text)
    print("encoded_input = ")
    for key, value in encoded_input.items():
        print(key, value, sep='\n', end='\n\n')

    input_tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'])
    print("input_tokens", input_tokens, sep='\n', end="\n\n")


def debug_gpt2():
    text = "Replace me by any text you'd like."
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
    encoded_input = tokenizer(text, return_tensors='pt')

    model:GPT2Model = GPT2Model.from_pretrained(gpt2_path)
    output:BaseModelOutputWithPastAndCrossAttentions = model.forward(**encoded_input)

    print(output.__dict__.keys())


if __name__ == "__main__":
    # debug_bpe()
    # debug_tokenize()
    # debug_tokenizer()
    debug_gpt2()
    ...