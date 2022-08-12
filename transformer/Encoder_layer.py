import tensorflow as tf

from transformer.MultiheadAttention import Attention
from transformer.Transformers_utils import FeedForwardNetwork


class EncoderBlock(tf.keras.layers.Layer):
    """The building block that makes the encoder stack of layers, consisting of an
    attention sublayer and a feed-forward sublayer.
    """
    def __init__(self, hidden_size, num_heads, filter_size, dropout_rate):
        """Constructor.
        Args:
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(EncoderBlock, self).__init__()
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._mha = Attention(hidden_size, num_heads)
        self._layernorm_mha = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._dropout_mha = tf.keras.layers.Dropout(dropout_rate)

        self._ffn = FeedForwardNetwork(hidden_size, filter_size)
        self._layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, padding_mask, training):
        """Computes the output of the encoder layer.
        Args:
            inputs: float tensor of shape [batch_size, src_seq_len, hidden_size],
                the input source sequences.
            padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
                populated with either 0 (for tokens to keep) or 1 (for tokens
                to be masked).
          training: bool scalar, True if in training mode.

        Returns:
          outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            output source sequences.
        """
        query = reference = self._layernorm_mha(inputs)
        outputs = self._mha(query, reference, padding_mask, training)
        ffn_inputs = self._dropout_mha(outputs, training=training) + inputs

        outputs = self._layernorm_ffn(ffn_inputs)
        outputs = self._ffn(outputs, training)
        outputs = self._dropout_ffn(outputs, training=training) + ffn_inputs
        return outputs

if __name__ == '__main__':
    from .Transformers_utils import get_padding_mask

    encoder = EncoderBlock(hidden_size=512, num_heads=8, filter_size=2048, dropout_rate=0.1)

    input = tf.random.uniform(shape=(1, 25, 512), dtype=tf.float32, minval=-10, maxval=10)
    input_sentence = tf.random.uniform(shape=(1, 25), dtype=tf.float32, minval=-10, maxval=10)
    enc_mask = get_padding_mask(input_sentence)

    output = encoder(input, enc_mask)
    print(output.shape)