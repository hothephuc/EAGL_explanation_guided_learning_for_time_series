cfg = {
    "data": {
        "db_dir": "F://EAGLe_explanation_attention_guided_learning_for_time_series//data//afib",
        "train": "train.csv",
        "test": "test.csv",
        "val": "val.csv",
    },
    
    "train" : {
        "num_classes": 9,
        "batch_size": 32,
        "num_workers": 4,
        "transform": None,
        "optimizer": "adam",
        "lr": 3e-4,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "epochs": 10,
        "log_interval": 10,
        "save_model": True,
        "save_model_path": "models/",
        "load_model": False,
        "load_model_path": "models/",
        "load_model_name": "resnet.pth"
    },

    "val":{
        "batch_size": 32,
        "num_workers": 4,
        "transform": None
    },

    "model": {
        "device": "cuda",
        "block": "BasicBlock",
        "num_blocks": [2, 2, 2, 2]
    },
}