import torch

from solar_forecasting.constants import DECODER_INPUT_FEATURES, ENCODER_FEATURES
from solar_forecasting.model import TimeSeriesTransformer


def test_transformer_output_shape():
    batch_size = 2
    lookback_steps = 8
    lookforward_steps = 4

    model = TimeSeriesTransformer(
        encoder_feature_dim=len(ENCODER_FEATURES),
        decoder_feature_dim=len(DECODER_INPUT_FEATURES),
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )

    src = torch.randn(batch_size, lookback_steps, len(ENCODER_FEATURES))
    tgt = torch.randn(batch_size, lookforward_steps, len(DECODER_INPUT_FEATURES))
    src_padding_mask = torch.zeros(batch_size, lookback_steps, dtype=torch.bool)
    tgt_padding_mask = torch.zeros(batch_size, lookforward_steps, dtype=torch.bool)

    output = model(src, tgt, tgt_padding_mask, src_padding_mask)

    assert output.shape == (batch_size, lookforward_steps, 1)
