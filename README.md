# Semi-supervised GAN

- This is an implementation of Semi-supervised generative adversarial network with fashion-mnist dataset.
- # Architecture
  <img src="https://github.com/ahmedkotb98/Semi-Supervised-Gan/blob/main/images/architecture.png" width="600" height="600" />

## Training

- You can use the below command to run the training on GPU
```
python ImprovedGAN.py --cuda
```

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
    - 1 - Hard to find “Nash equilibrium”
    This is the optimal point in the game for both generator G and discriminator D
    it’s really hard to find because this is a non coperative game where two players push each other as hard as possible.
    Nash equilibrium happens when one player does not change his/her actions regardless of what the other is doing.
    - 2 -  Vanishing gradient

        This is a very often problem we see in deep neural networks in general, the same problem gets stronger here because the gradient at Discriminator not only goes back to Discriminator network but also it goes back to Generator network as feedback.

        Because of it there is no stability in training GAN’s.

        →if the discriminator D gets stronger quickly (say D(x)= 1 , D(G(z)) =0 ), at generator G → log(1 — D(G(z))) = log(1–0) = 0 

        then the gradient of the loss function is 0 , then the learning is stopped.

        → if the discriminator D gets too weekly , then the generator G does not have good feed back so the loss represent nothing much.

        Moral: Don’t train D too good or too poor.
     
    - 3 - No proper evaluation metric

        As we have seen above in the code, we don’t really know when to stop the training as there is no proper evaluation metric in training GAN’s.

        Visual inspection is required, a lot of people do that it when training GAN’s.

        The losses don’t really tell much in GAN’s unlike other deep learning algorithms.

        Due to this, often we end up not having a good GAN model.

## Reference
- https://github.com/Sleepychord/ImprovedGAN-pytorch
- https://www.linkedin.com/pulse/ch14-general-adversarial-networks-gans-math-madhu-sanjeevi-mady-/


