# tacotron2_hifitts

Для запуска обучения необходимо:
1. Скачать модель (tacotron2)[https://github.com/NVIDIA/tacotron2]
2. Скачать (файлы для замены)[https://github.com/ViktorKrasnorutskiy/tacotron2_hifitts]
3. Заменить в ./tacotron2/ -> data_utils.py, hparams.py, model.py, train.py, utils.py
4. Скачать (zip-архив)[https://drive.google.com/file/d/1a0Xd6NxuErgdfsphZgZ7j_bUmGzZn5dX/view?usp=sharing] с подготовленным датасетом hifitts в виде мел-спектограмм.
   Либо обработать исходный датасет (hifitts)[http://www.openslr.org/109/] с помощью -> dataset_setup.ipynb.
5. Запустить ./tacotron2/train.py (>python train.py --output_directory=outdir --log_directory=logdir)

#### Готовая модель [https://drive.google.com/drive/folders/1cppLRoSbRlPSo5EhGc1lQQs2ADfKr3t8?usp=sharing] (нужно распаковать ./tacotron2/hifitts.zip)
