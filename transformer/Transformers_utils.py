import tensorflow as tf


def get_padding_mask(inputs, padding_value=0):
    """Creates a binary tensor to mask out padded tokens.
    Args:
        inputs: int tensor of shape [batch_size, src_seq_len], token ids
          of source sequences.
        padding_value: int scalar, the vocabulary index of the PAD token.
    Returns:
        mask: binary tensor of shape [batch_size, 1, 1, src_seq_len], storing ones
          for padded tokens and zeros for regular tokens.
    """
    mask = tf.cast(tf.equal(inputs, padding_value), 'float32')
    mask = mask[:, tf.newaxis, tf.newaxis, :]
    return mask


def get_look_ahead_mask(seq_len):
    """Creates a tensor to mask out future tokens in the target sequences when in
    training mode.
    Given sequence length `L` of target sequence, the mask would be a L x L
    matrix (when `tf.squeeze`'ed) where upper diagonal entries are ones and all
    other entries zeros.
        0, 1, 1, ..., 1
        0, 0, 1, ..., 1

          ... ...

        0, 0, 0, ..., 0
    Args:
        seq_len: int scalar tensor, sequence length.
    Returns:
        mask: float tensor of shape [1, 1, seq_len, seq_len], the mask tensor.
    """
    mask = 1 - tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
    mask = mask[tf.newaxis, tf.newaxis, :, :]
    return mask


def get_positional_encoding(seq_len, hidden_size, reverse=False):
    """Creates a tensor that encodes positional information.
    Args:
        seq_len: int scalar tensor, sequence length.
        hidden_size: int scalar, the hidden size of continuous representation.
        reverse: bool, whether to reverse the sequence. Defaults to False.
    Returns:
        positional_encoding: float tensor of shape [seq_len, hidden_size], the
          tensor that encodes positional information.
    """
    distances = tf.cast(tf.range(seq_len), 'float32')
    hidden_size //= 2
    inverse_frequencies = 1 / (
            10000 ** (tf.cast(tf.range(hidden_size), 'float32') / (hidden_size - 1)))
    positional_encoding = tf.einsum('i,j->ij', distances, inverse_frequencies)
    positional_encoding = tf.concat([tf.sin(positional_encoding),
                                     tf.cos(positional_encoding)], axis=1)
    return positional_encoding

class Projection(tf.keras.layers.Layer):
    """Linearly projects a batch of continuously represented sequences of tokens.
    This projection layer operates in either Split mode or Merge mode:
    - Split mode converts the input sequences in the original representation
      into the multi-headed "query", "key" or "value" for the attention
      computation.
      Input: [batch_size(N), seq_len(T), hidden_size(D)]
      Weight: [hidden_size(D), num_heads(H), size_per_head(S)]
      Output: dot([N*T, D], [D, H*S]) reshape ==> [N, T, H, S]
    - Merge mode performs the opposite action of Split, converting the
      multi-headed "value" back to the original representation.
      Input: [batch_size(N), seq_len(T), num_heads(H), size_per_head(S)]
      Weight: [num_heads(H), size_per_head(S), hidden_size(D)]
      Output: dot([N*T, H*S], [H*S, D]) reshape ==> [N, T, D]
    """
    def __init__(self,
               num_heads,
               size_per_head,
               kernel_initializer='glorot_uniform',
               mode="split"):
        """Constructor.
        Args:
          num_heads: int scalar, num of attention heads.
          size_per_head: int scalar, the hidden size of each attention head.
          kernel_initializer: string scalar, the weight initializer.
          mode: string scalar, mode of projection ("split" or "merge").
        """
        super(Projection, self).__init__()
        if mode not in ('split', 'merge'):
            raise ValueError('"mode" must be either "split" or "merge".')
        self._num_heads = num_heads
        self._size_per_head = size_per_head
        self._hidden_size = num_heads * size_per_head
        self._kernel_initializer = kernel_initializer
        self._mode = mode

    def build(self, inputs_shape):
        """Creates weights of this layer.
        Args:
          inputs_shape: tuple of ints or 1-D int tensor, the last element
            corresponds to the depth.
        """
        depth = inputs_shape[-1]
        if depth is None:
            raise ValueError('The depth of inputs must not be None.')

        if self._mode == 'merge':
            kernel_shape = self._num_heads, self._size_per_head, self._hidden_size
        else:
            kernel_shape = self._hidden_size, self._num_heads, self._size_per_head

        self.add_weight(name='kernel',
                        shape=kernel_shape,
                        initializer=self._kernel_initializer,
                        dtype='float32',
                        trainable=True)
        super(Projection, self).build(inputs_shape)

    def call(self, inputs):
        """Performs the projection.
        Args:
          inputs: float tensor of shape [batch_size, seq_len, num_heads,
            size_per_head] in Merge mode, or float tensor of shape [batch_size,
            seq_len, hidden_size] in Split mode.
            Returns:
              outputs: float tensor of shape [batch_size, seq_len, hidden_size] in
                Merge mode, or float tensor of shape [batch_size, seq_len, num_heads,
                size_per_head] int Split mode.
        """
        kernel = self.trainable_variables[0]
        if self._mode == 'merge':
            outputs = tf.einsum('NTHS,HSD->NTD', inputs, kernel)
        else:
            outputs = tf.einsum('NTD,DHS->NTHS', inputs, kernel)
        return outputs

class FeedForwardNetwork(tf.keras.layers.Layer):
    """The Projection layer that consists of a tandem of two dense layers (an
    intermediate layer and an output layer).
    """
    def __init__(self,
                 hidden_size,
                 filter_size,
                 dropout_rate=0.1,
                 filter_activation=tf.nn.relu):
        """Constructor.
        Args:
          hidden_size: int scalar, the hidden size of continuous representation,
            which is also the depth of the output dense layer.
          filter_size: int scalar, the depth of the intermediate dense layer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
          filter_activation: callable or string, activation function of the filter
            dense layer. Defaults to ReLU.
        """
        super(FeedForwardNetwork, self).__init__()
        self._hidden_size = hidden_size
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate
        self._filter_activation = filter_activation

        self._dense_layer_filter = tf.keras.layers.Dense(
            filter_size, use_bias=True, activation=filter_activation)
        self._dense_layer_output = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self._dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        """Performs projection through two dense layers.
        Args:
          inputs: float tensor of shape [batch_size, seq_len, hidden_size], the
            input sequences.
          training: bool scalar, True if in training mode.
        Return:
          outputs: float tensor of shape [batch_size, seq_len, hidden_size], the
            output sequences.
        """
        outputs = self._dense_layer_filter(inputs)
        outputs = self._dropout_layer(outputs, training=training)
        outputs = self._dense_layer_output(outputs)
        
        return outputs

if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    text = tokenizer.encode_plus("I have to go out", padding="max_length", max_length=20, return_tensors='tf',
                               truncation=True, add_special_tokens=True)

    inputs = text['input_ids']
    output_padding = get_padding_mask(inputs, padding_value=1)
    print("get_padding_mask: ", output_padding)

    print("get look_ahead_mask", get_look_ahead_mask(inputs.shape[1]))
    posi_enc = get_positional_encoding(seq_len=inputs.shape[1], hidden_size=512)
    print("get_positional_encoding shape: ", posi_enc.shape)

    ############################ TEST PROJECTION ############################
    project_split = Projection(num_heads=8, size_per_head=64, mode='split')
    input_split = tf.random.uniform((1, 25, 512), dtype=tf.float32, minval=-10, maxval=10)
    output_split = project_split(input_split, )
    print("output_split shape: ", output_split.shape)

    project_merge = Projection(num_heads=8, size_per_head=64, mode='merge')
    input_merge = tf.random.uniform((1, 25, 8, 64), dtype=tf.float32, minval=-10, maxval=10)
    output_merge = project_merge(input_merge, )
    print("output_merge shape: ", output_merge.shape)