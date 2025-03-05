import torch
def evaluate(model, data_loader, device):
    """
    Test a VAE model.

    Parameters:
    model: [VAE]
        The VAE model to test.
    data_loader: [torch.utils.data.DataLoader]
        The data loader to use for testing.
    device: [torch.device]
        The device to use for testing.
    """
    model.eval()

    with torch.no_grad():
        elbo_total = 0
        for x in data_loader:
            x = x[0].to(device)
            elbo = model.elbo(x)
            elbo_total += elbo.item()

    return elbo_total / len(data_loader)