import numpy as np

from rl.context import SubgoalEncoder


def test_subgoal_encoder_normalizes_output():
    encoder = SubgoalEncoder(dim=32)
    vec = encoder.encode("click the submit button")
    assert vec.shape == (32,)
    assert np.isclose(np.linalg.norm(vec), 1.0)
