import os
import json
import torch

from tokenization_glm import GLMGPT2Tokenizer, GLMBatchEncoding
from configuration_glm import GLMConfig
from modeling_glm import GLMPreTrainedModel, GLMForConditionalGeneration
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils import Trie

torch.set_printoptions(linewidth=500)  # 方便阅读

glm2b_path = os.path.join(os.getenv('my_data_dir'), "pretrained", "THUDM", "glm-2b")
glm10b_path = os.path.join(os.getenv('my_data_dir'), "pretrained", "THUDM", "glm-10b")


def debug_trie():
    """
    测试前缀树。https://en.wikipedia.org/wiki/Trie
    """
    trie = Trie()
    trie.add("Hello 友達")
    print(trie.data)

    trie.add("Hello")
    print(trie.data)

    
    trie = Trie()
    text = "[CLS] This is a extra_id_100"
    output = trie.split(text)
    print(output)

    trie.add("[CLS]") # 添加一个词 "[CLS]"
    trie.add("extra_id_1") # 添加一个词 "extra_id_1"
    trie.add("extra_id_100") # 添加一个词 "extra_id_100"，注意这个词的前缀"extra_id_1"也是一个词。
    # 切分按照，最长前缀切分
    output = trie.split(text)
    print(output)
    ...


def debug_token_cut():
    """
    调试 glm 的分词， glm 使用的是 gpt2的分词。
    """
    text = "Tsinghua University is located in [MASK]."

    tokenizer: GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    output = tokenizer.tokenize(text=text)
    print(output)


def debug_tokenizer():
    """
    调试tokenizer, 
    """
    # 数据
    train_data = ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"]
    targets = ["Beijing", "No"]

    tokenizer: GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs_pt: BatchEncoding = tokenizer(train_data, return_tensors="pt", padding=True)  # build_inputs_with_special_tokens

    # 
    print("tokenized = ")
    for key, value in inputs_pt.items():
        print(key, value, sep='\n', end='\n\n')

    # 句子开始  '[CLS]' 50259
    # 句子结束/填充 '<|endoftext|>' 50256  
    # 遮掩 '[MASK]' 50260
        
    input_ids_batched = tokenizer(train_data, padding=True)['input_ids']
    for input_ids in input_ids_batched:
        print("input_ids", input_ids, "input_tokens", tokenizer.convert_ids_to_tokens(input_ids), sep='\n', end='\n\n')

    
def debug_gen_inputs():
    """
    调试，生成式模型输入
    """
    # 数据
    train_data = ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"]
    targets = ["Beijing", "No"]

    tokenizer: GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs: BatchEncoding = tokenizer(train_data, return_tensors="pt", padding=True)  # build_inputs_with_special_tokens

    # 这个max_gen_length 是论文中 spans 的最大长度
    inputs: BatchEncoding = tokenizer.build_inputs_for_generation(inputs, targets=targets, max_gen_length=8, padding=False)

    print("model_inputs = ")
    for key, value in inputs.items():
        print(key, value, sep='\n', end='\n\n')
    ...

def debug_config():
    """
    调试config, 这个也可以在本地进行
    """
    config = GLMConfig.from_pretrained(glm2b_path)
    print("config = ", config, sep='\n', end='\n\n')

def debug_gen_model_forward():
    """
    调试glm 条件生成类的模型：GLMForConditionalGeneration
    """
    # 数据
    train_data = ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"]
    targets = ["Beijing", "No"]

    # 构造模型
    config = GLMConfig.from_pretrained(glm2b_path)
    model = GLMForConditionalGeneration.from_pretrained(glm2b_path, config=config)  # 模型默认加载到cpu上了。
    model.train()

    # 模型输入
    tokenizer: GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs = tokenizer(train_data, return_tensors="pt", padding=True)
    inputs = tokenizer.build_inputs_for_generation(inputs, targets=targets, max_gen_length=8, padding=False)

    # 移动到相同的设备上。
    inputs = inputs.to('cuda')  # 移动到gpu上。
    model = model.half().cuda()  # 移动到gpu上。可以使用半精度进一步降低显存使用，注意了：half() 在cpu上不支持。

    # 一次前向传播
    outputs = model.forward(**inputs)
    loss = outputs.loss
    logits = outputs.logits
    print("loss = ", loss, "logits = ", logits, sep='\n', end='\n\n')

def debug_gen_model_forward_small():
    """
    调试一个small版本的
    """
    # 数据
    train_data = ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"]
    targets = ["Beijing", "No"]

    # 构造模型
    config = GLMConfig.from_pretrained(glm2b_path)
    config.num_layers = 2  # 方便调试
    model = GLMForConditionalGeneration(config)
    model.train()

    # 模型输入
    tokenizer: GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs = tokenizer(train_data, return_tensors="pt", padding=True)
    inputs = tokenizer.build_inputs_for_generation(inputs, targets=targets, max_gen_length=8, padding=False)

    # 一次前向传播
    outputs = model.forward(**inputs)
    loss = outputs.loss
    logits = outputs.logits
    print("loss = ", loss, "logits = ", logits, sep='\n', end='\n\n')

def debug_gen_model_inferance():
    """
    调试，生成推理。
    """
    # 测试数据
    test_data = "Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai."

    # 模型
    config = GLMConfig.from_pretrained(glm2b_path)
    model = GLMForConditionalGeneration.from_pretrained(glm2b_path, config=config)
    model.eval()

    # 输入
    tokenizer: GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs = tokenizer(test_data, return_tensors="pt")
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)

    # 移动到相同的设备上。
    # inputs = inputs.to('cuda') # 移动到gpu上。
    # model = model.half().cuda() # 移动到gpu上。可以使用半精度进一步降低显存使用，注意了：half() 在cpu上不支持。

    # 模型推理
    outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
    print(tokenizer.decode(outputs[0].tolist()))

def debug_cls_inputs():
    """
    调试分类的模型输入
    """
    train_data = ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"]
    choices = [["Beijing", "Shanghai"], ["Yes", "No"]]

    tokenizer: GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs = tokenizer(train_data, return_tensors="pt", padding=True)
    inputs:GLMBatchEncoding = tokenizer.build_inputs_for_multiple_choice(inputs, choices)
    print("model_inputs = ")
    for key, value in inputs.items():
        print(key, value, sep='\n', end='\n\n')

    
    ...

if __name__ == "__main__":
    # debug_trie()
    debug_token_cut()
    # debug_tokenizer()

    # debug_gen_inputs()
    # debug_config()
    # debug_gen_model_forward()
    # debug_gen_model_forward_small()
    # debug_gen_model_inferance()
    
    # debug_cls_inputs()
    ...
