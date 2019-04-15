// jsonnet allows local variables like this
local char_embedding_dim = 128;
local hidden_dim = 128;
local num_epochs = 100;
local patience = 10;
local batch_size = 8;
local learning_rate = 0.1;

{
    "train_data_path": "../data/training.txt",
    "validation_data_path": "../data/validation.txt",
    "dataset_reader": {
        "type": "name-reader"
    },
    "model": {
        "type": "name-classifier",
        "name_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": char_embedding_dim
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": char_embedding_dim,
            "hidden_size": hidden_dim
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": [["name", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
        "patience": patience
    }
}
