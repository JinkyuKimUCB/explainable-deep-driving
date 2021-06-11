class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

MODE = "VA"
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
        "lr":           1e-3,
        "timelen":      4,
        "use_smoothing": None,
        "alpha":        1.0,    # coefficient for exp smoothing
        "UseFeat":      False,
        "maxiter":      60100,
        "save_steps":   5000,
        "val_steps":    10, #100,
        "model_path":   "./model/CNN/",
        "pretrained_model_path": None,
        "gpu_fraction": 0.8 }) 
elif MODE=="VA":
    config = dict2(**{
        "mode":         "VA",
        "dataset":      "BDD",
        "update_rule":  "adam",
        "CNNmodel":     "NVIDIA",
        "h5path":       "./data/processed/",
        "imgRow":       90,
        "imgCol":       160,
        "imgCh":        3,
        "resizeFactor": 1,
        "n_epoch":      100000,
        "epoch":        5, #20,
        "maxiter":      70100,
        "lr":           1e-4,
        "save_steps":   1000,
        "val_steps":    100,
        "model_path":   "./model/VA/",
        "pretrained_model_path": None,
        "test_replay":  None,
        "use_smoothing": None, #"Exp",
        "UseFeat":      True,
        "alpha":        1.0,
        "dim_hidden":   1024,
        "batch_size":   2,     #20,    
        "timelen":      20+3,  #20+3, 
        "ctx_shape":    [240,64],
        "alpha_c":      0.0})
else:
    raise NotImplementedError
