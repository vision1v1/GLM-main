import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
data_dir = os.getenv('my_data_dir')
pretrained_dir = os.path.join(data_dir, 'pretrained')

# tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
# model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True)


tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_dir, "THUDM", "glm-2b"), trust_remote_code=True)
exit()
model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(pretrained_dir, "THUDM", "glm-2b"), trust_remote_code=True)
model = model.half().cuda()
model.eval()

def debug_glm():
    """
    """
    
    ...


def debug_inference():
    # Inference
    test_data = "Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai."

    inputs = tokenizer(test_data, return_tensors="pt")
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    inputs = inputs.to('cuda')
    outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
    print(tokenizer.decode(outputs[0].tolist()))


def debug_training():
    # Training
    train_data = "Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"
    inputs = tokenizer([train_data], return_tensors="pt", padding=True)
    inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)
    inputs = inputs.to('cuda')
    outputs = model(**inputs)
    loss = outputs.loss
    logits = outputs.logits


if __name__ == "__main__":
    # debug_inference()
    debug_training()
    ...
