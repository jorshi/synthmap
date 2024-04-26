# synthmap

### Final project for ECS7022P - Computational Creativity at Queen Mary University of London

This project explores personalized latent spaces as macro controls for a synthesizer. VAEs are trained using synthesizer presets generated from a genetic algorithm evolving populations towards a target sound. The latent space is regularized using timbre features to provide control over synthesizer parameters within the vicinity of sound like the target.

For this project I am using a PyTorch implementation of a synthesizer inspired by the Roland TR-808 and look at sound matching and control of 808-like sounds.
