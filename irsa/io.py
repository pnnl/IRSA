import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from irsa.networks import PairedNeuralNet
from jcamp import JCAMP_reader
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from torch.utils.data import Dataset


def load_model(path, model=PairedNeuralNet, **kwargs):
    """
    Load model from path.

    Parameters
    ----------
    path : str
        Path to saved model state dict.
    model : :obj:`~torch.Module`
        Model class to initialize.
    kwargs
        Keyword arguments for model initialization.

    Returns
    -------
    :obj:`~torch.Module`
        Initialized model.

    """

    m = model(**kwargs)
    m.load_state_dict(torch.load(path))
    return m


def load_experimental(path):
    """
    Load experimental spectra from path using JCAMP.

    Parameters
    ----------
    path : str
        Path to DX file.

    Returns
    -------
    freq : :obj:`~numpy.array`
        Array of frequencies.
    intensity : :obj:`~numpy.array`
        Array of intensities.

    """

    # Load experimental spectra with JCAMP
    spec = JCAMP_reader(path)

    # Frequency
    freq = spec['x']

    # Intensity
    intensity = spec['y']

    # Normalize
    intensity = (intensity - intensity.min()) / \
        (intensity.max() - intensity.min())
    intensity = intensity / intensity.sum()

    # Data frame
    df = pd.DataFrame({'frequency': freq, 'intensity': intensity}).sort_values(
        by='intensity', ascending=False)

    # Drop duplicates
    df = df.drop_duplicates(subset='frequency').sort_values(
        by='frequency').reset_index(drop=True)

    return df['frequency'].values, df['intensity'].values


def load_predicted(path):
    """
    Load NWChem-predicted spectra from path using ISiCLE.

    Parameters
    ----------
    path : str
        Path to ISiCLE file.

    Returns
    -------
    freq : :obj:`~numpy.array`
        Array of frequencies.
    intensity : :obj:`~numpy.array`
        Array of intensities.

    """

    # Load predicted spectra
    with open(path, 'rb') as f:
        spec = pickle.load(f)

    # Check for completion
    if not hasattr(spec, 'frequency'):
        raise ValueError('Frequency attribute not detected in input.')

    # Frequency
    freq = np.abs(spec.frequency['frequencies'])

    # Intensity
    intensity = np.abs(spec.frequency['intensity arbitrary'])

    return freq.astype(np.float32), intensity.astype(np.float32)


def preprocess_predicted(freq, intensity, exp_freq, sigma=2):
    """
    Process predicted spectra to broaden, normalize, and map to experimental
    frequencies.

    Parameters
    ----------
    freq : :obj:`~numpy.array`
        Array of predicted frequencies.
    intensity : :obj:`~numpy.array`
        Array of predicted intensities.
    exp_freq : :obj:`~numpy.array`
        Array of experimental frequencies.
    sigma : int
        Sigma value to broaden by Gaussian function.

    Returns
    -------
    freq : :obj:`~numpy.array`
        Array of experimental frequencies.
    intensity : :obj:`~numpy.array`
        Array of processed intensities at experimental frequencies.

    """

    # Index of intersecting frequencies
    idx = (freq >= exp_freq.min()) & (freq <= exp_freq.max())

    # Truncate
    freq = freq[idx]
    intensity = intensity[idx]

    # Ensure both experimental and predicted points are sampled
    freq = np.concatenate((freq, exp_freq))
    intensity = np.concatenate((intensity, np.zeros_like(exp_freq)))

    # Ensure monotonic frequencies
    idx = np.argsort(freq)
    freq = freq[idx]
    intensity = intensity[idx]

    # Apply naive broadening function
    gauss = gaussian_filter1d(intensity, sigma=sigma)

    # Fit 1D spline to broadened spectra
    spl = interp1d(freq, gauss, kind='linear',
                   bounds_error=False, fill_value=0)

    # Sample spline at same points as experimental
    y_test = spl(exp_freq)

    # Normalize
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

    return exp_freq.copy().astype(np.float32), (y_test / y_test.sum()).astype(np.float32)


class PairedExpPredDataset(Dataset):

    def __init__(self, exp, exp_labels, pred, pred_labels, deterministic=False):
        self.exp = torch.from_numpy(exp.astype(np.float32))
        self.exp_labels = exp_labels
        self.pred = torch.from_numpy(pred.astype(np.float32))
        self.pred_labels = pred_labels
        self.deterministic = deterministic

        # Shape magic
        if len(self.exp.shape) != len(self.pred.shape):
            raise ValueError(
                'Shape mismatch between predicted and experimental')

        # (M, ) to (N, M) where N=1
        if len(self.exp.shape) == 1:
            self.exp = torch.unsqueeze(self.exp, 0)
            self.pred = torch.unsqueeze(self.pred, 0)

        # (N, M) to (N, K, M) where K=1
        if len(self.exp.shape) == 2:
            self.exp = torch.unsqueeze(self.exp, 1)
            self.pred = torch.unsqueeze(self.pred, 1)

        # Check if shape conversion successful
        if len(self.exp.shape) != 3:
            raise ValueError('Unable to coerce correct shape')

        # Boolean pairwise comparison matrix
        bmat = self.exp_labels[:, None] == self.pred_labels

        # Indices of nonmatched pairs
        self.neg_pairs = np.stack(np.where(bmat is False)).T

        # Shuffle if determinisitic
        if self.deterministic is True:
            np.random.shuffle(self.neg_pairs)

        # Indices of matched pairs
        self.pos_pairs = np.stack(np.where(bmat is True)).T

        # Labels
        self.pos_label = torch.from_numpy(np.array([1.0], dtype=np.float32))
        self.neg_label = torch.from_numpy(np.array([0.0], dtype=np.float32))

    def __len__(self):
        '''
        Length will be twice the number of matching instances, as half will
        be different by design.

        '''

        return 2 * self.pos_pairs.shape[0]

    def _get_same(self, idx):
        '''
        Return same-labeled spectra, label=1.

        '''

        # Get matching pair indices
        idx1, idx2 = self.pos_pairs[idx, :]

        # Release instance
        return OrderedDict([('exp_spec', self.exp[idx1]),
                            ('exp_label', self.exp_labels[idx1]),
                            ('pred_spec', self.pred[idx2]),
                            ('pred_label', self.pred_labels[idx2]),
                            ('label', self.pos_label)])

    def _get_different(self):
        '''
        Return different experimental and predicted data, label=0. As this class
        is larger, indices will be selected randomly.

        '''

        # Get non-matching pair indices
        idx1, idx2 = self.neg_pairs[np.random.randint(
            0, len(self.neg_pairs)), :]

        # Release instance
        return OrderedDict([('exp_spec', self.exp[idx1]),
                            ('exp_label', self.exp_labels[idx1]),
                            ('pred_spec', self.pred[idx2]),
                            ('pred_label', self.pred_labels[idx2]),
                            ('label', self.neg_label)])

    def _get_different_deterministic(self, idx):
        '''
        Return same-labeled spectra, label=1.

        '''

        # Get matching pair indices
        idx1, idx2 = self.neg_pairs[idx, :]

        # Release instance
        return OrderedDict([('exp_spec', self.exp[idx1]),
                            ('exp_label', self.exp_labels[idx1]),
                            ('pred_spec', self.pred[idx2]),
                            ('pred_label', self.pred_labels[idx2]),
                            ('label', self.neg_label)])

    def __getitem__(self, idx):
        '''
        Alternate same, different data.

        '''

        if idx % 2 == 0:
            return self._get_same(idx // 2)

        if self.deterministic is True:
            return self._get_different_deterministic(idx // 2)

        return self._get_different()
