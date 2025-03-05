import yaml
import argparse 
import os
import torch
import torch.nn as nn
import torch.distributions as td
from datetime import datetime
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from VAE.modules.decoders import BernoulliDecoder as Decoder
from VAE.modules.encoders import GaussianEncoder as Encoder
from VAE.modules.priors import GaussianPrior, MoGPrior, VampPrior
from VAE.modules.VAE import VAE
from VAE.modules.utils import evaluate, plot_combined_prior_posterior

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(model, optimizer, train_data_loader, test_data_loader, epochs, device,save_path):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    save_path: [str]
        The path to save the trained model(s).
    """
    model.train()

    total_steps = len(train_data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(train_data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

        # save, evaluate and sample the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{epoch+1}.pth"))
            # evaluate the model
            evaluation_elbo = evaluate(model, test_data_loader, device)
            # write test loss to a csv file
            csv_file = os.path.join(save_path, 'ELBO.csv')
            with open(csv_file, 'a') as file:
                file.write(f"Epoch {epoch+1}: ELBO: {evaluation_elbo:.4f}\n")
            print(f"Epoch {epoch+1}: ELBO: {evaluation_elbo:.4f}")
            # sample from the model
            with torch.no_grad():
                samples = (model.sample(64)).cpu() 
                save_image(samples.view(64, 1, 28, 28), os.path.join(save_path, f'samples_{epoch+1}.png'), nrow=8)
            # plot the prior and posterior
            plot_save_path = os.path.join(save_path, f'prior_posterior_{epoch+1}.png')
            plot_combined_prior_posterior(model, train_data_loader, device, plot_save_path)            
    
    # save the final model
    torch.save(model.state_dict(), os.path.join(save_path, "model_final.pth"))
    # evaluate the final model
    evaluation_elbo = evaluate(model, test_data_loader, device)
    # write test loss to a csv file
    csv_file = os.path.join(save_path, 'ELBO.csv')
    with open(csv_file, 'a') as file:
        file.write(f"Epoch {epochs}: ELBO: {evaluation_elbo:.4f}\n")
    print(f"Epoch {epochs}: ELBO: {evaluation_elbo:.4f}")
    # sample from the model
    with torch.no_grad():
        samples = (model.sample(64)).cpu() 
        save_image(samples.view(64, 1, 28, 28), os.path.join(save_path, 'samples_final.png'), nrow=8)
    # plot the prior and posterior
    plot_save_path = os.path.join(save_path, 'prior_posterior_final.png')
    plot_combined_prior_posterior(model, train_data_loader, device, plot_save_path)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='debug', help='Configuration file without extension')
    parser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility')
    args = parser.parse_args()
    config_file = os.path.join('VAE/configs', args.config + '.yaml')
    config = load_config(config_file)
    # ensure seed is set
    if args.seed is not None:
        config['seed'] = args.seed
    else:
        if 'seed' not in config:
            raise ValueError("Seed not set in config file or as argument")    

    # Access configuration values directly
    PRIOR_TYPE = config['prior']
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    LATENT_DIM = config['latent_dim']
    K = config['K']
    SEED = config['seed']

    # Try to use GPU if available else use CPU
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed
    torch.manual_seed(SEED)

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=BATCH_SIZE, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=BATCH_SIZE, shuffle=True)

    # Create the VAE model
    M = LATENT_DIM    

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )
    # Define VAE model
    decoder = Decoder(decoder_net)
    encoder = Encoder(encoder_net)
    if PRIOR_TYPE == 'gaussian':
        prior = GaussianPrior(M)
    elif PRIOR_TYPE == 'mog':
        prior = MoGPrior(M, K)
    elif PRIOR_TYPE == 'vamp':
        prior = VampPrior(M, encoder, K)
    else:
        raise ValueError(f"Unknown prior type: {PRIOR_TYPE}")

    # create folder with timestamp and save model and config
    os.makedirs('VAE/trained_models', exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # add the prior type, latent dim and seed to the timestamp
    timestamp = f"{PRIOR_TYPE}_{M}_{SEED}_{timestamp}"
    folder = os.path.join('VAE/trained_models', timestamp)
    os.makedirs(folder, exist_ok=True)    
    with open(os.path.join(folder, 'used_config.yaml'), 'w') as file:
        yaml.dump(config, file)
    model = VAE(prior, decoder, encoder).to(DEVICE)    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    
    train(model, optimizer, mnist_train_loader, mnist_test_loader, EPOCHS, DEVICE, folder)
    print(f"Training completed. Models saved in {folder}/model.pth") 