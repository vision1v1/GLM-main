from datasets import load_dataset

def load_bookcorpus():
    dataset = load_dataset("bookcorpus")
    print(dataset)



if __name__ == "__main__":
    load_bookcorpus()
    ...