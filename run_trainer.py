import tensorflow as tf
import pickle as pkl
import numpy

from absl import app
from absl import flags
from transformers import AutoTokenizer
from transformer.Transformer_Model import TransformerModel
from model_train import SequenceTransducerTrainer
from data_helper import make_dataset


flags.DEFINE_string(
    'path_pkl', 'translation/envi-nlp/En-Vi_data.pkl', 'Path to the directory data files (with '
        'pattern *train*) for training.')
flags.DEFINE_string(
    'model_dir', 'checkpoints/train', 'Path to the directory that checkpoint files will be '
        'written to.')
# model
flags.DEFINE_integer('encoder_stack_size', 6, 'Num of layers in encoder stack.')
flags.DEFINE_integer('decoder_stack_size', 6, 'Num of layers in decoder stack.')
flags.DEFINE_integer('hidden_size', 768, 'The dimensionality of the embedding vector.')
flags.DEFINE_integer('num_heads', 12, 'Num of attention heads.')
flags.DEFINE_integer('filter_size', 3072,
        'The depth of the intermediate dense layer of the feed-forward sublayer.')
flags.DEFINE_float('dropout_rate', 0.3, 'Dropout rate for the Dropout layers.')
#flags.DEFINE_integer('max_num_tokens', 4096, 'The maximum num of tokens in each batch.')
flags.DEFINE_integer('max_length', 128, 'Source or target seqs longer than'
                                       ' this will be filtered out.')

# training
#flags.DEFINE_integer('num_parallel_calls', 8,
#                     'Num of TFRecord files to be processed concurrently.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
#flags.DEFINE_integer('epochs', 30, 'Epochs')
flags.DEFINE_float('learning_rate', 2.0, 'Base learning rate.')
flags.DEFINE_float('lr_warmup_steps', 16000, 'Number of warm-ups steps.')
flags.DEFINE_float('adam_beta1', 0.9, '`beta1` of Adam optimizer.')
flags.DEFINE_float('adam_beta2', 0.997, '`beta2` of Adam optimizer.')
flags.DEFINE_float('adam_epsilon', 1e-9, '`epsilon` of Adam optimizer.')
flags.DEFINE_float('label_smoothing', 0.1, 'Amount of probability mass withheld for negative classes.')
flags.DEFINE_integer('num_steps', 300000, 'Num of training iterations (minibatches).')
flags.DEFINE_integer('save_ckpt_per_steps', 3000, 'Every this num of steps to save checkpoint.')
FLAGS = flags.FLAGS


def set_gpu(gpu_ids_list):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpus_used = [gpus[i] for i in gpu_ids_list]
            tf.config.set_visible_devices(gpus_used, 'GPU')

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""
    def __init__(self, learning_rate, hidden_size, warmup_steps):
        """Constructor.
        Args:
            learning_rate: float scalar, the base learning rate.
            hidden_size: int scalar, the hidden size of continuous representation.
            warmup_steps: int scalar, the num of warm-up steps
    """
        super(LearningRateSchedule, self).__init__()
        self._learning_rate = learning_rate
        self._hidden_size = hidden_size
        self._warmup_steps = tf.cast(warmup_steps, 'float32')

    def __call__(self, global_step):
        """Computes learning rate with linear warmup and rsqrt decay.
        Args:
            global_step: int scalar tensor, the current global step.
        Returns:
            learning_rate: float scalar tensor, the learning rate as a function of
            the input `global_step`.
        """
        global_step = tf.cast(global_step, 'float32')
        learning_rate = self._learning_rate
        learning_rate *= (self._hidden_size**-0.5)
        # linear warmup
        learning_rate *= tf.minimum(1.0, global_step / self._warmup_steps)
        # rsqrt decay
        learning_rate /= tf.sqrt(tf.maximum(global_step, self._warmup_steps))
        return learning_rate
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def main(_):
    path_pkl = FLAGS.path_pkl
    model_dir = FLAGS.model_dir

    encoder_stack_size = FLAGS.encoder_stack_size
    decoder_stack_size = FLAGS.decoder_stack_size
    hidden_size = FLAGS.hidden_size
    num_heads = FLAGS.num_heads
    filter_size = FLAGS.filter_size
    dropout_rate = FLAGS.dropout_rate

    max_length = FLAGS.max_length
    #epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    lr = FLAGS.learning_rate
    num_steps = FLAGS.num_steps
    lr_warmup_steps = FLAGS.lr_warmup_steps
    optimizer_adam_beta1 = FLAGS.adam_beta1
    optimizer_adam_beta2 = FLAGS.adam_beta2
    optimizer_adam_epsilon = FLAGS.adam_epsilon

    label_smoothing = FLAGS.label_smoothing
    
    save_ckpt_per_steps = FLAGS.save_ckpt_per_steps


    with open(path_pkl, 'rb') as f:
        train_pairs, val_pairs = pkl.load(f)
    
    # Processing for Vn
    train_pairs[1] = ['start ' + line  + ' end' for line in train_pairs[1]]
    val_pairs[1] = ['start ' + line  + ' end' for line in val_pairs[1]]
    print("\nProcessing for VN done!!!")
    # load Tokenizer
    VietTokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    EngTokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print("\nDownload tokenizer done!!!")
    # Tokenize text
    train_ds = make_dataset(train_pairs, EngTokenizer, VietTokenizer,
                            batch_size, max_length)
    val_ds = make_dataset(val_pairs, EngTokenizer, VietTokenizer,
                          batch_size, max_length)
    print("\n\nTokenizer done!!!")
        
    # get vocab size
    vocab_input = EngTokenizer.vocab_size
    vocab_target = VietTokenizer.vocab_size
    
    #get_batch = numpy.array(next(iter(val_ds))).shape[1]
    
    #steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    #ep = int(num_steps // steps_per_epoch)

    # transformers model
    model = TransformerModel(vocab_size_inp=vocab_input,
                             vocab_size_tgt=vocab_target,
                             encoder_stack_size=encoder_stack_size,
                             decoder_stack_size=decoder_stack_size,
                             hidden_size=hidden_size,
                             num_heads=num_heads,
                             filter_size=filter_size,
                             dropout_rate=dropout_rate,)

    # learning rate and optimizer
    learning_rate = LearningRateSchedule(lr, hidden_size, lr_warmup_steps)
    optimizer = tf.keras.optimizers.Adam( learning_rate,
                                          optimizer_adam_beta1,
                                          optimizer_adam_beta2,
                                          epsilon=optimizer_adam_epsilon)
    #learning_rate = CustomSchedule(hidden_size, lr_warmup_steps)
    #optimizer = tf.keras.optimizers.Adam(learning_rate, 
    #                                     beta_1=optimizer_adam_beta1, 
    #                                     beta_2=optimizer_adam_beta2, 
    #                                     epsilon=optimizer_adam_epsilon)

    # checkpoint
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    # calculated step training
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    epochs = int(num_steps // steps_per_epoch)
    
    #print("Step/Epochs: ", steps_per_epoch)
    #print("Num epoch: ", epochs)
    
    # build trainer and start training
    trainer = SequenceTransducerTrainer(model, label_smoothing)
    trainer.train(train_ds, val_ds, epochs, optimizer, ckpt, model_dir, num_steps, save_ckpt_per_steps)


if __name__ == '__main__':
    
    #set_gpu([0, 1])
    #flags.mark_flag_as_required('path_ckpt')
    #flags.mark_flag_as_required('model_dir')
    #flags.mark_flag_as_required('vocab_path')
    app.run(main)