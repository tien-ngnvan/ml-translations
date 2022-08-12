import tensorflow as tf

from transformer.Decoder_layer import DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Decoder that consists of a stack of structurally identical layers."""
    def __init__( self, stack_size, hidden_size, num_heads, filter_size, dropout_rate):
        """Constructor.
        Args:
          stack_size: int scalar, the num of layers in the stack.
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(Decoder, self).__init__()
        self._stack_size = stack_size
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._stack = [DecoderBlock(
            hidden_size, num_heads, filter_size, dropout_rate)
            for _ in range(self._stack_size)]
        self._layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._dropout = tf.keras.layers.Dropout(0.1)

    def call(self,
             inputs,
             encoder_outputs,
             look_ahead_mask,
             padding_mask,
             training,
             cache=None):
        """Computes the output of the decoder stack of layers.

        Args:
            inputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size],
            the input target sequences.
            encoder_outputs: float tensor of shape [batch_size, src_seq_len,
                hidden_size], the encoded source sequences to be used as reference.
            look_ahead_mask: float tensor of shape [1, 1, tgt_seq_len, tgt_seq_len],
                populated with either 0 (for tokens to keep) or 1 (for tokens
                to be masked).
            padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
                populated with either 0 (for tokens to keep) or 1 (for tokens
                to be masked).
            training: bool scalar, True if in training mode.
            cache: (Optional) dict with keys 'layer_0', ...
                'layer_[self.num_layers - 1]', where the value
            associated with each key is a dict with entries
              'k': tensor of shape [batch_size * beam_width, seq_len, num_heads,
                size_per_head],
              'v': tensor of shape [batch_size * beam_width, seq_len, num_heads,
                size_per_head],
              'tgt_tgt_attention': tensor of shape [batch_size * beam_width,
                num_heads, tgt_seq_len, tgt_seq_len],
              'tgt_src_attention': tensor of shape [batch_size * beam_width,
                num_heads, tgt_seq_len, src_seq_len].
            Must be provided in inference mode.

        Returns:
          outputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
            output target sequences.
        """
        for i, layer in enumerate(self._stack):
            inputs = layer.call(inputs,
                                encoder_outputs,
                                look_ahead_mask,
                                padding_mask,
                                training,
                                cache=cache['layer_%d' % i]
                                if cache is not None else None)
        outputs = self._dropout(self._layernorm(inputs))
        return outputs