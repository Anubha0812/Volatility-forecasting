# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pax ML model for patched time-series decoder.

The file implements Residual MLPs, Patched Decoder layers and PAX ML models.
"""

import dataclasses
import logging
from typing import Optional, Tuple

import einshape as es
from jax import lax, debug
import jax.numpy as jnp
from praxis import base_layer
from praxis import base_model
from praxis import layers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations
from praxis.layers import embedding_softmax
from praxis.layers import linears
from praxis.layers import normalizations
from praxis.layers import stochastics
from praxis.layers import transformers

# PAX shortcuts
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
template_field = base_layer.template_field

PAD_VAL = 1123581321.0
DEFAULT_QUANTILES = [0.00625,  0.0125, 0.01875, 0.025, 0.10, 0.25, 0.5, 0.75, 0.9]
DEFAULT_QUANTILE_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 1, 1]
DEFAULT_CROSSING_LAMBDA = 0.

# NestedMap keys
_INPUT_TS = "input_ts"
_TARGET_FUTURE = "actual_ts"
_INPUT_PADDING = "input_padding"
_OUTPUT_TS = "output_ts"
_FREQ = "freq"
_OUTPUT_TOKENS = "output_tokens"
_STATS = "stats"

# Small numerical value.
_TOLERANCE = 1e-7


def _shift_padded_seq(mask: JTensor, seq: JTensor) -> JTensor:
  """Shifts rows of seq based on the first 0 in each row of the mask."""
  num = seq.shape[1]

  # Find the index of the first 0 in each row of the mask
  first_zero_idx = jnp.argmin(mask, axis=1)

  # Create a range array for indexing
  idx_range = jnp.arange(num)

  def shift_row(carry, x):
    seq_row, shift = x
    shifted_idx = (idx_range - shift) % num
    shifted_row = seq_row[shifted_idx]
    return carry, shifted_row

  # Use lax.scan to shift each row of seq based on the corresponding
  # first_zero_idx.
  _, shifted_seq = lax.scan(shift_row, None, (seq, first_zero_idx))

  return shifted_seq


class ResidualBlock(base_layer.BaseLayer):
  """Simple feedforward block with residual connection.

  Attributes:
    input_dims: input dimension.
    hidden_dims: hidden dimension.
    output_dims: output dimension.
    dropout_prob: dropout probability.
    layer_norm: whether to use layer norm or not.
    dropout_tpl: config for dropout.
    ln_tpl: config for layer norm.
    act_tpl: config for activation in hidden layer.
  """

  input_dims: int = 0
  hidden_dims: int = 0
  output_dims: int = 0
  dropout_prob: float = 0.0
  layer_norm: bool = False
  dropout_tpl: LayerTpl = template_field(stochastics.Dropout)
  ln_tpl: LayerTpl = template_field(normalizations.LayerNorm)
  act_tpl: LayerTpl = template_field(activations.Swish)

  def setup(self):
    lnorm_tpl = self.ln_tpl.clone()
    lnorm_tpl.dim = self.output_dims
    self.create_child("ln_layer", lnorm_tpl)

    dropout_tpl = self.dropout_tpl.clone()
    dropout_tpl.keep_prob = 1.0 - self.dropout_prob
    self.create_child("dropout", dropout_tpl)

    self.create_child(
        "hidden_layer",
        pax_fiddle.Config(
            linears.FeedForward,
            input_dims=self.input_dims,
            output_dims=self.hidden_dims,
            activation_tpl=self.act_tpl.clone(),
        ),
    )

    self.create_child(
        "output_layer",
        pax_fiddle.Config(
            linears.FeedForward,
            input_dims=self.hidden_dims,
            output_dims=self.output_dims,
            activation_tpl=pax_fiddle.Config(activations.Identity),
        ),
    )

    self.create_child(
        "residual_layer",
        pax_fiddle.Config(
            linears.FeedForward,
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            activation_tpl=pax_fiddle.Config(activations.Identity),
        ),
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    hidden = self.hidden_layer(inputs)
    output = self.output_layer(hidden)
    output = self.dropout(output)
    residual = self.residual_layer(inputs)
    if self.layer_norm:
      return self.ln_layer(output + residual)
    else:
      return output + residual


def _masked_mean_std(inputs: JTensor,
                     padding: JTensor) -> Tuple[JTensor, JTensor]:
  """Calculates mean and standard deviation of arr across axis 1.

  It should exclude values where pad is 1.

  Args:
    inputs: A JAX array of shape [b, n, p].
    padding: A JAX array of shape [b, n, p] with values 0 or 1.

  Returns:
    A tuple containing the mean and standard deviation of arr. We return the
    statistics of the first patch with more than three non-padded values.
  """
  # Selecting the first pad with more than 3 unpadded values.
  pad_sum = jnp.sum(1 - padding, axis=2)

  def _get_patch_index(arr: JTensor):
    indices = jnp.argmax(arr >= 3, axis=1)
    row_sum = (arr >= 3).sum(axis=1)
    return jnp.where(row_sum == 0, arr.shape[1] - 1, indices)

  patch_indices = _get_patch_index(pad_sum)
  bidxs = jnp.arange(inputs.shape[0])

  arr = inputs[bidxs, patch_indices, :]
  pad = padding[bidxs, patch_indices, :]

  # Create a mask where P is 0
  mask = 1 - pad

  # Calculate the number of valid elements
  num_valid_elements = jnp.sum(mask, axis=1)

  num_valid_elements = jnp.where(num_valid_elements == 0, 1, num_valid_elements)

  # Calculate the masked sum and squared sum of M
  masked_sum = jnp.sum(arr * mask, axis=1)
  masked_squared_sum = jnp.sum((arr * mask)**2, axis=1)

  # Calculate the masked mean and standard deviation
  masked_mean = masked_sum / num_valid_elements
  masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
  masked_var = jnp.where(masked_var < 0.0, 0.0, masked_var)
  masked_std = jnp.sqrt(masked_var)

  return masked_mean, masked_std


def _create_quantiles() -> list[float]:
  """Returns the quantiles for forecasting."""
  return DEFAULT_QUANTILES

def _create_quantile_weights() -> list[float]:
  """Returns quantile weights for forecasting."""
  return DEFAULT_QUANTILE_WEIGHTS


class PatchedTimeSeriesDecoder(base_layer.BaseLayer):
  """Patch decoder layer for time-series foundation model.

  Attributes:
    patch_len: length of input patches.
    horizon_len: length of output patches. Referred to as `output_patch_len`
      during inference.
    model_dims: model dimension of stacked transformer layer.
    hidden_dims: hidden dimensions in fully connected layers.
    quantiles: list of quantiles for non prob model.
    residual_block_tpl: config for residual block.
    stacked_transformer_params_tpl: config for stacked transformer.
    use_freq: whether to use frequency encoding.

  In all of what followed, except specified otherwise, B is batch size, T is
  sequence length of time-series. N is the number of input patches that can be
  obtained from T. P is the input patch length and H is the horizon length. Q is
  number of output logits. D is model dimension.
  """

  patch_len: int = 0
  horizon_len: int = 0
  model_dims: int = 0
  hidden_dims: int = 0
  quantiles: list[float] = dataclasses.field(default_factory=_create_quantiles)
  quantile_weights: list[float] = dataclasses.field(default_factory=_create_quantile_weights)
  crossing_lambda: float = dataclasses.field(default=DEFAULT_CROSSING_LAMBDA)
  residual_block_tpl: LayerTpl = template_field(ResidualBlock)
  stacked_transformer_params_tpl: LayerTpl = template_field(
      transformers.StackedTransformer)
  use_freq: bool = True
  use_pos_emb: bool = True

  def setup(self) -> None:
    """Construct the model."""
    num_outputs = len(self.quantiles) + 1

    stl = self.stacked_transformer_params_tpl.clone()
    stl.model_dims = self.model_dims
    stl.hidden_dims = self.hidden_dims
    stl.mask_self_attention = True

    self.create_child("stacked_transformer_layer", stl)

    input_resl = self.residual_block_tpl.clone()
    ff_in_dims = 2 * self.patch_len
    input_resl.input_dims = ff_in_dims
    input_resl.hidden_dims = self.hidden_dims
    input_resl.output_dims = self.model_dims
    self.create_child(
        "input_ff_layer",
        input_resl,
    )

    horizon_resl = self.residual_block_tpl.clone()
    horizon_resl.input_dims = self.model_dims
    horizon_resl.hidden_dims = self.hidden_dims
    horizon_resl.output_dims = self.horizon_len * num_outputs
    self.create_child(
        "horizon_ff_layer",
        horizon_resl,
    )

    self.create_child(
        "position_emb",
        pax_fiddle.Config(layers.PositionalEmbedding,
                          embedding_dims=self.model_dims),
    )

    if self.use_freq:
      self.create_child(
          "freq_emb",
          pax_fiddle.Config(
              embedding_softmax.Embedding,
              num_classes=3,
              input_dims=self.model_dims,
          ),
      )

  def transform_decode_state(
      self, transform_fn: base_layer.DecodeStateTransformFn) -> None:
    """Transforms all decode state variables based on transform_fn."""
    self.stacked_transformer_layer.transform_decode_state(transform_fn)

  def _forward_transform(
      self, inputs: JTensor,
      patched_pads: JTensor) -> Tuple[JTensor, Tuple[JTensor, JTensor]]:
    """Input is of shape [B, N, P]."""
    mu, sigma = _masked_mean_std(inputs, patched_pads)
    sigma = jnp.where(sigma < _TOLERANCE, 1.0, sigma)
    # Normalize each patch.
    outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
    outputs = jnp.where(
        jnp.abs(inputs - PAD_VAL) < _TOLERANCE, PAD_VAL, outputs)
    return outputs, (mu, sigma)

  def _reverse_transform(self, outputs: JTensor,
                         stats: Tuple[JTensor, JTensor]) -> JTensor:
    """Output is of shape [B, N, P, Q]."""
    mu, sigma = stats
    return outputs * sigma[:, None, None, None] + mu[:, None, None, None]

  def _preprocess_input(
      self,
      input_ts: JTensor,
      input_padding: JTensor,
      pos_emb: Optional[JTensor] = None,
  ) -> Tuple[JTensor, JTensor, Optional[Tuple[JTensor, JTensor]], JTensor]:
    """Preprocess input for stacked transformer."""
    # Reshape into patches.
    patched_inputs = es.jax_einshape("b(np)->bnp", input_ts, p=self.patch_len)
    patched_pads = es.jax_einshape("b(np)->bnp",
                                   input_padding,
                                   p=self.patch_len)
    patched_inputs = jnp.where(
        jnp.abs(patched_pads - 1.0) < _TOLERANCE, 0.0, patched_inputs)
    patched_pads = jnp.where(
        jnp.abs(patched_inputs - PAD_VAL) < _TOLERANCE, 1, patched_pads)
    patched_inputs, stats = self._forward_transform(patched_inputs,
                                                    patched_pads)

    # B x N x D
    patched_inputs = patched_inputs * (1.0 - patched_pads)
    concat_inputs = jnp.concatenate([patched_inputs, patched_pads], axis=-1)
    model_input = self.input_ff_layer(concat_inputs)
    # A patch should not be padded even if there is at least one zero.
    patched_padding = jnp.min(patched_pads, axis=-1)

    if self.use_pos_emb:
      if pos_emb is None:
        position_emb = self.position_emb(seq_length=model_input.shape[1])
      else:
        position_emb = pos_emb
      if self.do_eval:
        if position_emb.shape[0] != model_input.shape[0]:
          position_emb = jnp.repeat(position_emb, model_input.shape[0], axis=0)
        position_emb = _shift_padded_seq(patched_padding, position_emb)
      model_input += position_emb

    return model_input, patched_padding, stats, patched_inputs

  def _postprocess_output(
      self,
      model_output: JTensor,
      num_outputs: int,
      stats: Tuple[JTensor, JTensor],
  ) -> JTensor:
    """Postprocess output of stacked transformer."""
    # B x N x (H.Q)
    output_ts = self.horizon_ff_layer(model_output)
    output_ts = es.jax_einshape("bn(hq)->bnhq",
                                output_ts,
                                q=num_outputs,
                                h=self.horizon_len)
    return self._reverse_transform(output_ts, stats)

  def __call__(self, inputs: NestedMap) -> NestedMap:
    """PatchTST call.

    Args:
      inputs: A NestedMap containing (1) input_ts: input sequence of shape [B,
        T] where T must be multiple of patch_length; (2) input_padding: that
        contains padding map.

    Returns:
      A nested map with two keys:
      (1) 'output_tokens' of shape [B, N, D].
      (2) 'output_ts' of shape [B, N, H, Q]
      (3) 'stats' a Tuple of statistics for renormalization.
    """
    input_ts, input_padding = inputs[_INPUT_TS], inputs[_INPUT_PADDING]
    num_outputs = len(self.quantiles) + 1
    model_input, patched_padding, stats, _ = self._preprocess_input(
        input_ts=input_ts,
        input_padding=input_padding,
    )
    if self.use_freq:
      freq = inputs[_FREQ].astype(jnp.int32)
      f_emb = self.freq_emb(freq)  # B x 1 x D
      f_emb = jnp.repeat(f_emb, model_input.shape[1], axis=1)
      model_input += f_emb
    model_output = self.stacked_transformer_layer(model_input, patched_padding)

    output_ts = self._postprocess_output(model_output, num_outputs, stats)
    return NestedMap({
        _OUTPUT_TOKENS: model_output,
        _OUTPUT_TS: output_ts,
        _STATS: stats
    })

  def decode(
      self,
      inputs: NestedMap,
      horizon_len: int,
      output_patch_len: Optional[int] = None,
      max_len: int | None = None,
      return_forecast_on_context: bool = False,
  ) -> tuple[JTensor, JTensor]:
    """Auto-regressive decoding without caching.

    Args:
      inputs: input time-series and paddings. Time-series shape B x C, padding
        shape shape B x (C + H) where H is the prediction length.
      horizon_len: prediction length.
      output_patch_len: output length to be fetched from one step of
        auto-regressive decoding.
      max_len: maximum training context length.
      return_forecast_on_context: whether to return the model forecast on the
        context except the first input patch.

    Returns:
      Tuple of two forecasting results:
      - Point (mean) output predictions as a tensor with shape B x H'.
      - Full predictions (mean and quantiles) as a tensor with shape
        B x H' x (1 + # quantiles).
      In particular, if return_forecast_on_context is True, H' is H plus
      the forecastable context length, i.e. context_len - (first) patch_len.
    """
    final_out = inputs[_INPUT_TS]
    context_len = final_out.shape[1]
    paddings = inputs[_INPUT_PADDING]
    if max_len is None:
      max_len = context_len
    if self.use_freq:
      freq = inputs[_FREQ].astype(jnp.int32)
    else:
      freq = jnp.zeros([final_out.shape[0], 1], dtype=jnp.int32)
    full_outputs = []
    if paddings.shape[1] != final_out.shape[1] + horizon_len:
      raise ValueError(
          "Length of paddings must match length of input + horizon_len:"
          f" {paddings.shape[1]} != {final_out.shape[1]} + {horizon_len}")
    if output_patch_len is None:
      output_patch_len = self.horizon_len
    num_decode_patches = (horizon_len + output_patch_len -
                          1) // output_patch_len
    for step_index in range(num_decode_patches):
      current_padding = paddings[:, 0:final_out.shape[1]]
      input_ts = final_out[:, -max_len:]
      input_padding = current_padding[:, -max_len:]
      model_input = NestedMap(
          input_ts=input_ts,
          input_padding=input_padding,
          freq=freq,
      )
      fprop_outputs = self(model_input)[_OUTPUT_TS]
      if return_forecast_on_context and step_index == 0:
        # For the first decodings step, collect the model forecast on the
        # context except the unavailable first input batch forecast.
        new_full_ts = fprop_outputs[:, :-1, :self.patch_len, :]
        new_full_ts = es.jax_einshape("bnph->b(np)h", new_full_ts)

        full_outputs.append(new_full_ts)

      # (full batch, last patch, output_patch_len, index of mean forecast = 0)
      new_ts = fprop_outputs[:, -1, :output_patch_len, 0]
      new_full_ts = fprop_outputs[:, -1, :output_patch_len, :]
      # (full batch, last patch, output_patch_len, all output indices)
      full_outputs.append(new_full_ts)
      final_out = jnp.concatenate([final_out, new_ts], axis=-1)

    if return_forecast_on_context:
      # `full_outputs` indexing starts at after the first input patch.
      full_outputs = jnp.concatenate(full_outputs,
                                     axis=1)[:, :(context_len - self.patch_len +
                                                  horizon_len), :]
    else:
      # `full_outputs` indexing starts at the forecast horizon.
      full_outputs = jnp.concatenate(full_outputs, axis=1)[:, 0:horizon_len, :]

    return (full_outputs[:, :, 0], full_outputs)


class PatchedDecoderFinetuneModel(base_model.BaseModel):
  """Model class for finetuning patched time-series decoder.

  Attributes:
    core_layer_tpl: config for core layer.
    freq: freq to finetune on.
  """

  core_layer_tpl: LayerTpl = template_field(PatchedTimeSeriesDecoder)
  freq: int = 0

  def setup(self) -> None:
    self.create_child("core_layer", self.core_layer_tpl)

  def compute_predictions(self, input_batch: NestedMap) -> NestedMap:
    input_ts = input_batch[_INPUT_TS]
    input_padding = jnp.zeros_like(input_ts)
    context_len = input_ts.shape[1]
    input_patch_len = self.core_layer_tpl.patch_len
    context_pad = ((context_len + input_patch_len - 1) //
                   input_patch_len) * input_patch_len - context_len

    input_ts = jnp.pad(input_ts, [(0, 0), (context_pad, 0)])
    input_padding = jnp.pad(input_padding, [(0, 0), (context_pad, 0)],
                            constant_values=1)
    freq = jnp.ones([input_ts.shape[0], 1], dtype=jnp.int32) * self.freq
    new_input_batch = NestedMap(
        input_ts=input_ts,
        input_padding=input_padding,
        freq=freq,
    )
    return self.core_layer(new_input_batch)

  def _weighted_quantile_loss(
     self,
     preds: JTensor,
     actual: JTensor,
     quantiles: JTensor,
     quantile_weights: JTensor,
     reduction: str = "mean"
  ):
    """Calculates weighted quantile loss.
    
    Args:
        pred: B x T x Q
        actual: B x T
        quantiles: Q
        quantile_weights: Q
    
    Returns:
        aggregated quantile loss
    """
    # canonicalize actual -> [B, T], preds -> [B, T, Q]
    if preds.ndim == 2:  # [B, Q] -> [B, 1, Q]
        preds = preds[:, None, :]
    B, H, Q = preds.shape
    
    if actual.ndim == 1:
        actual = actual[:, None]  # [B] -> [B, 1]
    actual = actual.reshape(B, H)  # [B,T]
    
    taus = jnp.asarray(quantiles, dtype=preds.dtype).reshape((Q,))
    if taus.shape[0] != Q:
        raise ValueError("quantiles length must match Q")
    
    # u = y - y_pred, shape [B,T,Q]
    u = jnp.expand_dims(actual, axis=-1) - preds
    indicator = (u < 0).astype(preds.dtype)  # [B,T,Q]
    per_q_loss = (taus - indicator) * u  # pinball loss, >=0
    
    q_w = jnp.asarray(quantile_weights, dtype=preds.dtype).reshape((Q,))
    if q_w.shape[0] != Q:
        raise ValueError("quantile_weights length must match Q")
    
    # apply quantile weights and reduce over quantiles -> per-example per-horizon loss [B,T]
    per_example = jnp.sum(per_q_loss * q_w.reshape((1, 1, Q)), axis=-1)

    if reduction == "per_quantile":
      # mean across batch and horizon for each quantile
      per_q_mean = jnp.mean(per_q_loss, axis=(0, 1))  # [Q]
      return per_q_mean

    if reduction == "none":
      return per_example  # [B,T]
    elif reduction == "per_batch":
      return jnp.mean(per_example, axis=1) # [B]
    elif reduction == "sum":
      return jnp.sum(per_example)
    elif reduction == "mean":
      norm = jnp.sum(q_w)
      return jnp.sum(per_example) / jnp.maximum(norm, 1e-12)
    else:
      raise ValueError("reduction must be 'none','sum' or 'mean'")

  def _compute_weighted_quantile_loss(self, preds: JTensor, actual: JTensor):
    quantiles = jnp.asarray(self.core_layer.quantiles)
    q_weights = jnp.asarray(self.core_layer.quantile_weights, dtype=preds.dtype)
    qloss = self._weighted_quantile_loss(preds, actual, quantiles, quantile_weights=q_weights, reduction="per_batch")
    # Double loss if uniform weights - similar to the original implementation of qloss
    qloss =  qloss + qloss * jnp.array_equal(q_weights, jnp.ones_like(q_weights))
    return qloss

  def _quantile_crossing_penalty(self, preds: JTensor):
    """Calculates average quantile crossing as mean((relu(preds_q - preds_{q+1}))^2) - mean squared hinge loss.

    Args:
      preds: B x T x Q

    Returns:
      percentage of quantiles crossing
    """
    crossing_lambda = self.core_layer.crossing_lambda
    diffs = preds[..., :-1] - preds[..., 1:] # [B,T,Q-1]
    violations = jnp.maximum(diffs, 0.0)
    penalty = jnp.mean(jnp.square(violations), axis=(1, 2)) # [B]
    return crossing_lambda * penalty

  def compute_loss(
    self, prediction_output: NestedMap,
    input_batch: NestedMap
  ) -> Tuple[NestedMap, NestedMap]:
    output_ts = prediction_output[_OUTPUT_TS]
    actual_ts = input_batch[_TARGET_FUTURE]
    pred_ts = output_ts[:, -1, 0:actual_ts.shape[1], :]
    # compute per example losses for improved logging later
    per_example_mse = jnp.mean(jnp.square(pred_ts[:, :, 0] - actual_ts), axis=1) # [B]
    per_example_qloss = self._compute_weighted_quantile_loss(pred_ts[:, :, 1:], actual_ts) # [B]
    per_example_crossing_penalty = self._quantile_crossing_penalty(pred_ts[:, :, 1:]) # [B]
    # ensure same dtype (avoid float64 vs float32 mismatch)
    dtype = per_example_mse.dtype
    per_example_qloss = jnp.asarray(per_example_qloss, dtype=dtype)
    # aggregate for training loss
    mse_loss = jnp.mean(per_example_mse)
    quantile_loss = jnp.mean(per_example_qloss)
    crossing_penalty = jnp.mean(per_example_crossing_penalty)
    loss = mse_loss + quantile_loss + crossing_penalty
    # stop gradient for logging values
    metrics_list = [
      lax.stop_gradient(per_example_mse), lax.stop_gradient(per_example_qloss), lax.stop_gradient(per_example_crossing_penalty)
    ]
    # print([arr.shape for arr in metrics_list])
    metrics_arr = jnp.stack(metrics_list, axis=1) # [B,3]
    per_example_out = NestedMap(metrics=metrics_arr)

    loss_weight = jnp.array(1.0, dtype=jnp.float32)
    return {"avg_qloss": (loss, loss_weight)}, per_example_out
