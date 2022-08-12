import tensorflow as tf
from transformer.Transformers_utils import Projection


NEG_INF = -1e9

class Attention(tf.keras.layers.Layer):
    """Multi-headed attention.
    Given a batch of vector-represented query sequences (tensor of shape [
    batch_size, q_seq_len, hidden_size]) and context sequences (tensor of shape
    [batch_size, c_seq_len, hidden_size]), this layer computes a new
    representation of the query sequences by making them discriminatively attend
    to tokens in the context sequences.
    If the query and context happen to be the same, the result ends up being
    "Self Attention" -- the query sequence attends to itself.
    """
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        """Constructor.
        Args:
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(Attention, self).__init__()
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate
        self._size_per_head = hidden_size // num_heads

        self._dense_layer_query = Projection(
            num_heads, self._size_per_head, mode='split')
        self._dense_layer_key = Projection(
            num_heads, self._size_per_head, mode='split')
        self._dense_layer_value = Projection(
            num_heads, self._size_per_head, mode='split')
        self._dense_layer_output = Projection(
            num_heads, self._size_per_head, mode='merge')
        self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, query, context, attention_mask, training=False, cache=None):
        """Computes new representation of query sequences.
        Args:
          query: float tensor of shape [batch_size, q_seq_len, hidden_size],
            query sequences.
          context: float tensor of shape [batch_size, c_seq_len, hidden_size]
            , context sequences.
          attention_mask: float tensor of shape [batch_size, num_heads, q_seq_len,
            c_seq_len], populated with either 0 (for tokens to keep) or 1 (for
            tokens to be masked).
          training: (Optional) bool scalar, True if in training mode.
          cache: (Optional) dict with entries
            'k': tensor of shape [batch_size * beam_width, seq_len, num_heads,
              size_per_head],
            'v': tensor of shape [batch_size * beam_width, seq_len, num_heads,
              size_per_head],
            'tgt_tgt_attention': tensor of shape [batch_size * beam_width,
              num_heads, tgt_seq_len, tgt_seq_len],
            'tgt_src_attention': tensor of shape [batch_size * beam_width,
              num_heads, tgt_seq_len, src_seq_len].
            Must be provided in inference mode when called within decoder layers.
        Returns:
          outputs: float tensor of shape [batch_size, q_seq_len, hidden_size], the
            new representation of `query`.
        """
        self_attention = True if id(query) == id(context) else False

        # [batch_size, q_seq_len, num_heads, size_per_head]
        q = self._dense_layer_query(query)

        # [batch_size, c_seq_len, num_heads, size_per_head]
        k = self._dense_layer_key(context)
        v = self._dense_layer_value(context)

        if cache is not None and self_attention:
            # concatenate along the `seq_len` dimension
            cache['k'] = k = tf.concat([cache['k'], k], axis=1)
            cache['v'] = v = tf.concat([cache['v'], v], axis=1)

        # [batch_size, num_heads, q_seq_len, c_seq_len]
        attention_weights = tf.einsum('NQHS,NCHS->NHQC', q, k)
        attention_weights *= self._size_per_head ** -0.5
        attention_weights += attention_mask * NEG_INF
        attention_weights = tf.nn.softmax(attention_weights, axis=3)
        attention_weights = self._dropout_layer(
            attention_weights, training=training)

        # save attention weights of encoder layers in inference mode
        if not training and cache is None and self_attention:
            setattr(self, '_attention_weights', attention_weights)

        # save attention weights for visualization in inference mode
        if cache is not None:
            if self_attention:
                # [batch_size, num_heads, tgt_seq_len, tgt_seq_len]
                cache['tgt_tgt_attention'] = tf.concat([tf.pad(
                    cache['tgt_tgt_attention'], [[0, 0], [0, 0], [0, 0], [0, 1]]),
                    attention_weights], axis=2)
            else:
                # [batch_size, num_heads, tgt_src_len, src_seq_len]
                cache['tgt_src_attention'] = tf.concat([
                    cache['tgt_src_attention'], attention_weights], axis=2)

        # [batch_size, q_seq_len, num_heads, size_per_head]
        outputs = tf.einsum('NHQC,NCHS->NQHS', attention_weights, v)

        # [batch_size, q_seq_len, hidden_size]
        outputs = self._dense_layer_output(outputs)
        return outputs

if __name__ == '__main__':
    from .Transformers_utils import get_look_ahead_mask


    query = reference = tf.random.uniform(shape=(1, 25, 512), dtype=tf.float32, minval=-10, maxval=10)
    attention_mask = get_look_ahead_mask(query.shape[1])

    attn = Attention(hidden_size=512, num_heads=8, dropout_rate=0.1)
    output = attn(query, reference, attention_mask, training=False, cache=None)

    print(output.shape)