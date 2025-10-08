import numpy as np

from rl.features import DomTextHasher


def test_dom_encoder_dimension():
    encoder = DomTextHasher(dim=128)
    obs = {"dom_text": "Click the submit button"}
    vec = encoder.encode(obs)
    assert vec.shape == (128,)
    assert np.isclose(np.linalg.norm(vec), 1.0)
