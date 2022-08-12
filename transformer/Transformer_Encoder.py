import tensorflow as tf

from transformer.Encoder_layer import EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """The Encoder that consists of a stack of structurally identical layers."""
    def __init__( self, stack_size, hidden_size, num_heads, filter_size, dropout_rate):
        """Constructor.
        Args:
            stack_size: int scalar, num of layers in the stack.
            hidden_size: int scalar, the hidden size of continuous representation.
            num_heads: int scalar, num of attention heads.
            filter_size: int scalar, the depth of the intermediate dense layer of the
                feed-forward sublayer.
            dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(Encoder, self).__init__()
        self._stack_size = stack_size
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._stack = [EncoderBlock(hidden_size,
                                    num_heads,
                                    filter_size,
                                    dropout_rate) for _ in range(self._stack_size)]
        self._layernorm = tf.keras.layers.LayerNormalization()
        self._dropout = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, padding_mask, training):
        """Computes the output of the encoder stack of layers.
        Args:
            inputs: float tensor of shape [batch_size, src_seq_len, hidden_size],
                the input source sequences.
            padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
                        populated with either 0 (for tokens to keep) or 1 (for tokens to be
                        masked).
            training: bool scalar, True if in training mode.
       Returns:
         outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
           output source sequences.
       """
        for layer in self._stack:
            inputs = layer.call(inputs, padding_mask, training)
        outputs = self._layernorm(inputs)
        outputs = self._dropout(outputs)
        return outputs