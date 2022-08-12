import tensorflow as tf

def scce_with_ls(y_true, y_pred):
    y = tf.one_hot(tf.cast(y_true, tf.int32), vocab_target)

    cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    loss = cross_entropy(y, y_pred)
    
    mask = tf.logical_not(tf.math.equal(y_true, 1))  # output 0 for y=1 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
 
    loss = mask * loss
    loss = tf.reduce_mean(loss)

    return loss

def compute_loss(labels, logits, smoothing, vocab_size, padding_value=1):
    """Computes average (per-token) cross entropy loss.
    1. Applies label smoothing -- all entries in the groundtruth label tensor
        get non-zero probability mass.
    2. Computes per token loss of shape [batch_size, tgt_seq_len], where padded
        positions are masked, and then the sum of per token loss is normalized by
        the total number of non-padding entries.
    Args:
        labels: int tensor of shape [batch_size, tgt_seq_len], the groundtruth
            token ids.
        logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
            predicted logits of tokens over the vocabulary.
        smoothing: float scalar, the amount of label smoothing applied to the
            one-hot class labels.
        vocab_size: int scalar, num of tokens (including SOS and EOS) in the
            vocabulary.
        padding_value: int scalar, the vocabulary index of the PAD token.
    Returns:
        loss: float scalar tensor, the per-token cross entropy
    """
    # effective_vocab = vocab - {SOS_ID}
    effective_vocab_size = vocab_size - 1

    # prob mass allocated to the token that should've been predicted
    on_value = 1.0 - smoothing
    # prob mass allocated to all other tokens
    off_value = smoothing / (effective_vocab_size - 1)

    # [batch_size, tgt_seq_len, vocab_size]
    labels_one_hot = tf.one_hot(
        labels,
        depth=vocab_size,
        on_value=on_value,
        off_value=off_value)

    # compute cross entropy over all tokens in vocabulary but SOS_ID (i.e. 0)
    # because SOS_ID should never appear in the decoded sequence
    # [batch_size, tgt_seq_len]
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot[:, :, 1:], logits=logits[:, :, 1:])

    # this is the entropy when the softmax'ed logits == groundtruth labels
    # so it should be deducted from `cross_entropy` to make sure the minimum
    # possible cross entropy == 0
    normalizing_constant = -(on_value * tf.math.log(on_value) +
                             (effective_vocab_size - 1) * off_value * tf.math.log(off_value + 1e-20))
    cross_entropy -= normalizing_constant

    # mask out predictions where the labels == `padding_value`
    weights = tf.cast(tf.not_equal(labels, padding_value), 'float32')
    cross_entropy *= weights
    loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(weights)
    return loss