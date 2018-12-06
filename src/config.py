class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

MODE = "CNN"
if MODE=="CNN":
    config = dict2(**{
        "mode":         "CNN",
        "dataset":      "BDD",
        "update_rule":  "adam",
        "CNNmodel":     "NVIDIA",
        "h5path":       "./data/processed/",
        "imgRow":       90,
        "imgCol":       160,
        "imgCh":        3,
        "resizeFactor": 1,
        "batch_size":   200, 
        "skipvalidate": False,
        "lr":           1e-3,
        "timelen":      4,
        "use_smoothing": None,
        "alpha":        1.0,    # coefficient for exp smoothing
        "UseFeat":      False,
        "maxiter":      60100,
        "save_steps":   5000,
        "val_steps":    10, #100,
        "model_path":   "./model/CNN/",
        "pretrained_model_path": None }) 
else:
    raise NotImplementedError
