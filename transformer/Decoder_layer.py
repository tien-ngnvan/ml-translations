import tensorflow as tf

from transformer.MultiheadAttention import Attention
from transformer.Transformers_utils import FeedForwardNetwork


class DecoderBlock(tf.keras.layers.Layer):
    """The building block that makes the decoder stack of layers, consisting of a
    self-attention sublayer, cross-attention sublayer and a feed-forward sublayer.
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
        super(DecoderBlock, self).__init__()
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._mha_intra = Attention(hidden_size, num_heads)
        self._layernorm_mha_intra = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._dropout_mha_intra = tf.keras.layers.Dropout(dropout_rate)

        self._mha_inter = Attention(hidden_size, num_heads)
        self._layernorm_mha_inter = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._dropout_mha_inter = tf.keras.layers.Dropout(dropout_rate)

        self._ffn = FeedForwardNetwork(hidden_size, filter_size)
        self._layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self,
             inputs,
             encoder_outputs,
             look_ahead_mask,
             padding_mask,
             training,
             cache=None):
        """Computes the output of the decoder layer.

          Args:
            inputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
              input target sequences.
            encoder_outputs: float tensor of shape [batch_size, src_seq_len,
              hidden_size], the encoded source sequences to be used as reference.
            look_ahead_mask: float tensor of shape [1, 1, tgt_seq_len, tgt_seq_len],
              populated with either 0 (for tokens to keep) or 1 (for tokens to be
              masked).
            padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
              populated with either 0 (for tokens to keep) or 1 (for tokens to be
              masked).
            training: bool scalar, True if in training mode.
            cache: (Optional) dict with entries
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
        query = reference = self._layernorm_mha_intra(inputs)
        outputs = self._mha_intra(
            query, reference, look_ahead_mask, training, cache=cache)
        mha_inter_inputs = self._dropout_mha_intra(outputs, 
                                                   training=training) + inputs

        query, reference = self._layernorm_mha_inter(mha_inter_inputs
                                                    ), encoder_outputs
        outputs = self._mha_inter(
            query, reference, padding_mask, training, cache=cache)
        ffn_inputs = self._dropout_mha_inter(outputs, 
                                             training=training) + mha_inter_inputs

        outputs = self._layernorm_ffn(ffn_inputs)
        outputs = self._ffn(outputs, training)
        outputs = self._dropout_ffn(outputs, training=training) + ffn_inputs

        return outputs

if __name__ == '__main__':
    from .Transformers_utils import get_look_ahead_mask, get_padding_mask
    # encoder
    enc_output = tf.random.uniform(shape=(1, 25, 512), dtype=tf.float32, minval=-10, maxval=10)
    input_sentence = tf.random.uniform(shape=(1, 25), dtype=tf.float32, minval=-10, maxval=10)
    enc_mask = get_padding_mask(input_sentence)
    # decoder
    decoder = DecoderBlock(hidden_size=512, num_heads=8, filter_size=2048, dropout_rate=0.3)
    input_sentence = tf.random.uniform(shape=(1, 35), dtype=tf.float32, minval=-10, maxval=10)
    input_embds = tf.random.uniform(shape=(1, 35, 512), dtype=tf.float32, minval=-10, maxval=10)
    ahead_mask_dec = get_look_ahead_mask(input_sentence.shape[1])

    dec_out = decoder(input_embds, encoder_outputs=enc_output, look_ahead_mask=ahead_mask_dec, padding_mask=enc_mask,
                      training=True, cache=None)

    print(dec_out.shape)
