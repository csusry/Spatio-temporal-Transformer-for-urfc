# Spatio-temporal Transformer

Dataset Link: https://aistudio.baidu.com/datasetdetail/12529 

1. Create a new folder named `data` in the current directory and unzip the downloaded data into it. The `data` folder should contain three subfolders: `npy`, `train`, and `test`. The `npy` folder stores the preprocessed `train_visit` and `test_visit`, while the `train` and `test` folders contain the image data respectively. (Data preprocessing includes simple translation, rotation, and denoising, as well as the removal of all-black and all-white images, defogging, and histogram equalization beforehand.)

2. Install the necessary runtime libraries by executing `pip install -r requirements.txt`.

3. To start training and testing, run `python multimain.py`. Some hyperparameters such as `epoch`, `batch_size`, etc., can be modified in `config.py`.
