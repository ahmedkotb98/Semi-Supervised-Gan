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

## How to run the app

After installing necessary packages, use the following command to run the app from project root directory-
  
```
uvicorn app.main:app
```
And visit http://127.0.0.1:8000/docs from your browser. You will be able to see swagger. From there you can upload an image through predict endpoint and then you will get a json response.

<img src="https://github.com/ahmedkotb98/Semi-Supervised-Gan/blob/main/images/api_docs.png" width="1000" height="1000" />

## How to run the app with docker

Make sure you are in the project root directory and you have started docker. Then create docker image using the following command.

```
docker-compose up --build
```

## Some Notes

- If you want to use my models for inference please make sure that images like fashion-mnist images
- why Improving GAN’s training is a very hot research topic ? 
  1 - Hard to find “Nash equilibrium”
  This is the optimal point in the game for both generator G and discriminator D
  it’s really hard to find because this is a non coperative game where two players push each other as hard as possible.
  Nash equilibrium happens when one player does not change his/her actions regardless of what the other is doing.
