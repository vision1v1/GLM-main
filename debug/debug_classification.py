import os
from transformers import AutoTokenizer, AutoModelForMultipleChoice
data_dir = os.getenv('my_data_dir')
pretrained_dir = os.path.join(data_dir, 'pretrained')


def debug_inference():
    """
    调试分类任务，选择
    """
    
    train_data = ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"]
    choices = [["Beijing", "Shanghai"], ["Yes", "No"]]
    
    path = os.path.join(pretrained_dir, 'THUDM', 'glm-10b')

    # 模型
    model = AutoModelForMultipleChoice.from_pretrained(path, trust_remote_code=True)
    model.eval()

    # 输入
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    inputs = tokenizer(train_data, return_tensors="pt", padding=True)
    inputs = tokenizer.build_inputs_for_multiple_choice(inputs, choices)
    
    # 移动到相同设备
    model = model.half().cuda()
    inputs = inputs.to('cuda')

    # 推理
    outputs = model(**inputs)
    logits = outputs.logits


if __name__ == "__main__":
    ...
