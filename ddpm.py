# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from DDPM.modules.DDPM import DDPM

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

        if (epoch + 1) % 5 == 0:
            # Save model
            torch.save(model.state_dict(), f'ddpm_model_{epoch+1}.pt')
    

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='mnist', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    #parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    #parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--network', type=str, default='unet', choices=['fc', 'unet'], help='network architecture to use (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    if args.data == 'cb' or args.data == 'tg':
        from FLOW.modules.ToyData import TwoGaussians, Chequerboard

        n_data = 10000000
        toy = {'tg': TwoGaussians, 'cb': Chequerboard}[args.data]()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)

    elif args.data == 'mnist':
        from torchvision import transforms
        transform = transforms.Compose ([ transforms.ToTensor () ,
        transforms.Lambda (lambda x : x + torch.rand (x.shape ) /255) ,
        transforms.Lambda ( lambda x : (x -0.5) *2.0) ,
        transforms.Lambda ( lambda x : x.flatten () ) ])
        train_data = datasets.MNIST ('data', train =True , download =True , transform = transform )

        train_loader = torch.utils.data.DataLoader (train_data ,
                                                    batch_size = args.batch_size,
                                                    shuffle =True)


    # Get the dimension of the dataset
    #D = next(iter(train_loader)).shape[1]

    if args.network == 'fc':
        from DDPM.modules.nn_architectures import FcNetwork
        if args.data=='mnist':
            D = 784
        else:
            D = next(iter(train_loader)).shape[1]
        # Define the network
        num_hidden = 64
        network = FcNetwork(D, num_hidden)
    
    elif args.network == 'unet':  
        from DDPM.modules.nn_architectures import Unet           
        network = Unet()

    # Set the number of steps in the diffusion process
    T = 1000

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model = DDPM(network, T=T).to(DEVICE)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, DEVICE)

        # Save model
        torch.save(model.state_dict(), args.model)    

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

        # Generate samples
        model.eval()
        if args.data == 'tg' or args.data == 'cb':    
            with torch.no_grad():
                samples = (model.sample((10000,D))).cpu()            
            # Transform the samples back to the original space
            samples = samples /2 + 0.5

            # Plot the density of the toy data and the model samples
            coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(f"{args.model}_samples.png")
            plt.close()

        elif args.data == 'mnist':
            with torch.no_grad():
                num_samples = 100
                samples = (model.sample((num_samples,784))).cpu()
                samples = samples.view(num_samples,1,28,28)
                # transform the samples back to the original space
                samples = samples /2 + 0.5
                save_image(samples, f"{args.model}_samples.png", nrow=10)

