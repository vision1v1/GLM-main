import os
import json

from tokenization_glm import GLMGPT2Tokenizer
from configuration_glm import GLMConfig
from modeling_glm import GLMPreTrainedModel, GLMForConditionalGeneration

train_data = ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"]
test_data = "Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai."

glm2b_path = os.path.join(os.getenv('my_data_dir'), "pretrained", "THUDM", "glm-2b")

def debug_tokenizer():
    """
    调试tokenizer, 这个可以在本地调试
    """
    tokenizer:GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs = tokenizer(train_data, return_tensors="pt", padding=True) # build_inputs_with_special_tokens
    print("inputs = ", inputs, sep='\n')
    inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)
    print("inputs = ", inputs, sep='\n')


def debug_config():
    """
    调试config, 这个也可以在本地进行
    """
    config = GLMConfig.from_pretrained(glm2b_path)
    print("config = ", config)


def debug_model_forward():
    """
    调试glm 条件生成类的模型：GLMForConditionalGeneration
    """
    # 构造模型
    config = GLMConfig.from_pretrained(glm2b_path)
    model = GLMForConditionalGeneration.from_pretrained(glm2b_path, config=config) # 模型默认加载到cpu上了。
    model.train()
    
    # 模型输入
    tokenizer:GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs = tokenizer(train_data, return_tensors="pt", padding=True)
    inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)

    # 移动到相同的设备上。
    # inputs = inputs.to('cuda') # 移动到gpu上。
    # model = model.half().cuda() # 移动到gpu上。可以使用半精度进一步降低显存使用，注意了：half() 在cpu上不支持。

    # 一次前向传播
    outputs = model.forward(**inputs)
    loss = outputs.loss
    logits = outputs.logits
    print("loss = ", loss, "logits = ", logits, sep='\n', end='\n\n')


def debug_model_inferance():
    """
    """
    config = GLMConfig.from_pretrained(glm2b_path)
    model = GLMForConditionalGeneration.from_pretrained(glm2b_path, config=config)
    model.eval()

    tokenizer:GLMGPT2Tokenizer = GLMGPT2Tokenizer.from_pretrained(glm2b_path)
    inputs = tokenizer(test_data, return_tensors="pt")
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    
    # 移动到相同的设备上。
    # inputs = inputs.to('cuda') # 移动到gpu上。
    # model = model.half().cuda() # 移动到gpu上。可以使用半精度进一步降低显存使用，注意了：half() 在cpu上不支持。
    
    # 模型推理
    outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
    print(tokenizer.decode(outputs[0].tolist()))


if __name__ == "__main__":
    # debug_tokenizer()
    # debug_config()
    # debug_model_forward()
    debug_model_inferance()
    ...