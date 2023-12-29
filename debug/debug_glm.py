from transformers import AutoTokenizer

def load_glm2b():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-2b", trust_remote_code=True)
    print(type(tokenizer))

if __name__ == "__main__":
    load_glm2b()
    ...