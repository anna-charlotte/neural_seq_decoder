import torch

from neural_decoder.phoneme_utils import CHANNEL_ORDER
from neural_decoder.transforms import ReorderChannelTransform


def test_reorder_channel_transform():
    # create a tensor of shape (256, 1) with values ranging from 0 to 255
    column_tensor = torch.arange(256).unsqueeze(1)
    # repeat this tensor to create a tensor of shape (256, 32)
    sample_tensor = column_tensor.repeat(1, 32)
    assert sample_tensor.size() == (256, 32)

    # apply the transform
    transform = ReorderChannelTransform()
    result = transform(sample_tensor)

    assert result.size(), sample_tensor.size()

    expected_order = torch.tensor(CHANNEL_ORDER + [i + 128 for i in CHANNEL_ORDER])

    for i in range(32):
        assert result[:, i].size(0) == 256
        assert torch.allclose(result[:, i], expected_order)
