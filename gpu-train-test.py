tensorflow_not_loadable = False
pytorch_not_loadable = False

try:
    import tensorflow as tf
except:
    tensorflow_not_loadable = True

try:
    import torch
except:
    pytorch_not_loadable = True

if __name__ == '__main__':
    if tensorflow_not_loadable:
        print("[FATAL] [Tensorflow] Package not importable.")
        print("[FATAL] [Tensorflow] Skip testing.")
    else:
        t1 = tf.config.list_physical_devices('GPU')
        t2 = tf.test.is_built_with_cuda()

        if t1 and t2:
            print("[Tensorflow] Ok")
            print("[Tensorflow] Success loading CUDA")
            print("[Tensorflow] Training with CUDA available")
        else:
            print("[FATAL] [Tensorflow] GPU not loaded, or it is unavailable to train with.")
    
    if pytorch_not_loadable:
        print("[FATAL] [PyTorch] Package not importable.")
        print("[FATAL] [PyTorch] Skip testing.")
    else:
        t1 = torch.cuda.is_available()

        if t1:
            print("[PyTorch] Ok")
            print("[PyTorch] Success loading CUDA")
            print("[PyTorch] Training with CUDA available")
        else:
            print("[FATAL] [PyTorch] GPU not loaded, or it is unavailable to train with.")
    
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())

        print("[PyTorch] Successful tests with", device_name)