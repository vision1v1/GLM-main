from datasets import load_dataset

def load_bookcorpus():
    dataset = load_dataset("bookcorpus")
    print(dataset)
    train_dataset = dataset["train"]
    print(train_dataset)

    print(train_dataset[0:10])



if __name__ == "__main__":
    load_bookcorpus()
    ...