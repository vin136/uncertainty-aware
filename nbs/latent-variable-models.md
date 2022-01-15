# Introduction to Latent Variable Models

Autoregressive models are slow to sample as we assume all the variables dependent on each other. Similarly, Flow based models assume same dimensionality for the latent distribution. **Idea**:We can make part of the observation space independent conditioned on some latent variable.

![vae-1](https://user-images.githubusercontent.com/21222766/147595001-12db8d28-30fc-43d9-9384-7dac523cb23d.jpeg)


![vae-2](https://user-images.githubusercontent.com/21222766/147594976-cd3ad3b4-3348-4a55-a077-6900f8df5884.jpeg)

![vae-3](https://user-images.githubusercontent.com/21222766/147594980-827caf98-f570-4cdc-b333-76939c96ad3e.jpeg)

Now, We understand all the key mathematical ideas behind VAE's. Below I share two different derivations for `VLB`. These add further insights into the model.

![vae-4](https://user-images.githubusercontent.com/21222766/147699075-b532cdbe-0ea6-4578-a671-d99a91a52b90.jpeg)

![vae-5](https://user-images.githubusercontent.com/21222766/147699083-c96b3208-68f0-4748-8e98-ba467ecfb6ba.jpeg)

![vae-7](https://user-images.githubusercontent.com/21222766/147835220-b47315d0-1b45-4072-a381-576894637169.jpg)
