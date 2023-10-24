from llama_index import SimpleDirectoryReader
import json

def llama_index_get_text():
    documents = SimpleDirectoryReader(input_dir='documents', recursive=True).load_data()
    
    for i in documents:
        doc = nlp_es(i.text)
        text = " ".join([token.text for token in doc])
        # print(text)
        yield translate_es_en(text), i.hash
        
    del documents
    
def multi_xscience_get_text():
    
    labels = ["train", "test", "val"]
    
    dataset = {
        "train_x": [],
        "train_y": [],
        
        "test_x": [],
        "test_y": [],
        
        "val_x": [],
        "val_y": []
    }
    for i in labels:
        with open(f'documents/Multi-XScience/{i}.json') as f:
            data = json.loads(f.read())
    
        for item in data:
            new_dict = {}
            related = ""

            for key, value in item.items():

                if key == "related_work":
                    related = value
                else:
                    new_dict[key] = value
                    
            dataset[f"{i}_x"].append(new_dict)
            dataset[f"{i}_y"].append(related)

    train, test, val = ((dataset["train_x"], dataset["train_y"]),(dataset["test_x"], dataset["test_y"]),(dataset["val_x"], dataset["val_y"]),)
    return train,test,val            