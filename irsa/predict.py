import torch


def predict(model, exp, pred):
    """
    Helper function to predict similarity between experimental and predicted
    spectra given a trained model. Will attempt to coerce input shapes.

    Parameters
    ----------
    model : instance of PyTorch model
    exp : :obj:`~numpy.array`
        Array of experimental intensities. Shape (N, M).
    pred : :obj:`~numpy.array`
        Array of predicted intensities. Shape (N, M).

    Returns
    -------
    :obj:`~numpy.array`
        Array of predicted scores. Shape (N, ).

    """
    # Select device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Send to device
    model = model.to(device)

    # Model in evaluation mode
    model.eval()

    # Convert to tensor
    exp = torch.from_numpy(exp)
    pred = torch.from_numpy(pred)

    # Shape magic
    if len(exp.shape) != len(pred.shape):
        raise ValueError('Shape mismatch between predicted and experimental')

    # (M, ) to (N, M) where N=1
    if len(exp.shape) == 1:
        exp = torch.unsqueeze(exp, 0)
        pred = torch.unsqueeze(pred, 0)

    # (N, M) to (N, K, M) where K=1
    if len(exp.shape) == 2:
        exp = torch.unsqueeze(exp, 1)
        pred = torch.unsqueeze(pred, 1)

    # Check if shape conversion successful
    if len(exp.shape) != 3:
        raise ValueError('Unable to coerce correct shape')

    # Send to device
    exp = exp.to(device)
    pred = pred.to(device)

    # Predict
    with torch.no_grad():
        return torch.sigmoid(model(exp, pred)).detach().cpu().numpy().flatten()
