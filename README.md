# Semi-supervised GAN

- This is an implementation of Semi-supervised generative adversarial network with fashion-mnist dataset.
- # Architecture
  <img src="https://github.com/ahmedkotb98/Semi-Supervised-Gan/blob/main/images/architecture.png" width="400" height="400" />

## Training
 
- You can use `python ImprovedGAN.py --cuda` to run it on GPU

## Results

- after training you can achieve an accuracy higer than 90% on test dataset with 100 labeled data(10 per class) and other 59,000 unlabeled data after 200 epochs.

## Models

- Generator : https://drive.google.com/file/d/1-aGDeOglIPQtw-i4-Scwjpinf2YwJqC7/view?usp=sharing

- Discriminator : https://drive.google.com/file/d/1-ZJ5RPqHlCtVKDlH5z4urGRne8PDpibE/view?usp=sharing

## Deployment

- to run the app  
```
uvicorn app.main:app
```
-

