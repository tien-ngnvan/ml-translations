"""Translates source sequences into target sequences, and optionally evaluates 
BLEU score if groundtruth target sequences are provided.
"""
import tensorflow as tf
from absl import app
from absl import flags

from transformers import AutoTokenizer
from transformer.Transformer_Model import TransformerModel
from model_eval import SequenceTransducerEvaluator
from data_helper import process_data, make_dataset

flags.DEFINE_integer(
    'encoder_stack_size', 9, 'Num of layers in encoder stack.')
flags.DEFINE_integer(
    'decoder_stack_size', 9, 'Num of layers in decoder stack.')
flags.DEFINE_integer(
    'hidden_size', 512, 'The dimensionality of the embedding vector.')
flags.DEFINE_integer(
    'num_heads', 8, 'Num of attention heads.')
flags.DEFINE_integer(
    'filter_size', 2048, 'The depth of the intermediate dense layer of the'
        'feed-forward sublayer.')
flags.DEFINE_float(
    'dropout_rate', 0.3, 'Dropout rate for the Dropout layers.')
flags.DEFINE_integer(
    'extra_decode_length', 50, 'The max decode length would be'
        ' the sum of `tgt_seq_len` and `extra_decode_length`.')
flags.DEFINE_integer(
    'beam_width', 4, 'Beam width for beam search.')
flags.DEFINE_float(
    'alpha', 0.6, 'The parameter for length normalization used in beam search.')
flags.DEFINE_integer(
    'decode_batch_size', 32, 'Number of sequences in a batch for inference.')
flags.DEFINE_integer(
    'src_max_length', 128, 'The number of tokens that source sequences will be '
        'truncated or zero-padded to in inference mode.') 


flags.DEFINE_string(
    'source_text_filename', 'translation/envi-nlp/test.en', 'Path to the source text sequences to be '
        'translated.')
flags.DEFINE_string(
    'target_text_filename', 'translation/envi-nlp/test.vi', 'Path to the target (reference) text '
        'sequences that the translation will be checked against,')

flags.DEFINE_string(
    'translation_output_filename', 'translations.txt', 'Path to the output '
        'file that the translations will be written to.')
flags.DEFINE_string(
    'model_dir', 'checkpoints/train', 'Path to the directory that checkpoint files will be '
        'written to.')

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


def main(_):
    model_dir = FLAGS.model_dir

    encoder_stack_size = FLAGS.encoder_stack_size
    decoder_stack_size = FLAGS.decoder_stack_size
    hidden_size = FLAGS.hidden_size
    num_heads = FLAGS.num_heads
    filter_size = FLAGS.filter_size
    dropout_rate = FLAGS.dropout_rate

    extra_decode_length = FLAGS.extra_decode_length
    beam_width = FLAGS.beam_width
    alpha = FLAGS.alpha
    decode_batch_size = FLAGS.decode_batch_size
    src_max_length = FLAGS.src_max_length

    source_text_filename = FLAGS.source_text_filename
    target_text_filename = FLAGS.target_text_filename
    translation_output_filename = FLAGS.translation_output_filename
    
    # processing data test
    with open(source_text_filename, 'r', encoding='utf-8') as f:
        source_text = f.readlines()
    with open(target_text_filename, 'r', encoding='utf-8') as f:
        target_text = f.readlines()
        
    lines_en = [process_data(text) for text in source_text]
    lines_vi = [process_data(text) for text in target_text]
    test_pairs = [lines_en, lines_vi]
    test_pairs[1] = ['start ' + line  + ' end' for line in test_pairs[1]]
    
    # load Tokenizer
    VietTokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    EngTokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Tokenize text
    test_ds = make_dataset(test_pairs, EngTokenizer, VietTokenizer, 
                           decode_batch_size, 128)

    # get vocab size
    vocab_input = EngTokenizer.vocab_size
    vocab_target = VietTokenizer.vocab_size
    
    # transformers model
    model = TransformerModel(vocab_size_inp=vocab_input,
                             vocab_size_tgt=vocab_target,
                             encoder_stack_size=encoder_stack_size,
                             decoder_stack_size=decoder_stack_size,
                             hidden_size=hidden_size,
                             num_heads=num_heads,
                             filter_size=filter_size,
                             dropout_rate=dropout_rate,
                             extra_decode_length=extra_decode_length,
                             beam_width=beam_width,
                             alpha=alpha)
    
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')
    
    # build evaluator
    evaluator = SequenceTransducerEvaluator(model, VietTokenizer)
    translations, sorted_indices = evaluator.translate(test_ds)
    print(translations)
###########################################################################
    # translates input sequences, and optionally evaluates BLEU score if 
    # groundtruth target sequences are provided
    #if target_text_filename is not None:
    #    case_insensitive_score, case_sensitive_score = evaluator.evaluate(
    #        source_text_filename, target_text_filename)
    #    print('BLEU(case insensitive): %f' % case_insensitive_score)
    #    print('BLEU(case sensitive): %f' % case_sensitive_score)
    #else:
    #    evaluator.translate(
    #        source_text_filename, translation_output_filename)
    #    print('Inference mode: no groundtruth translations.\nTranslations written '
    #        'to file "%s"' % translation_output_filename)


if __name__ == '__main__':
    set_gpu([1])
    flags.mark_flag_as_required('source_text_filename')
    flags.mark_flag_as_required('model_dir')
    app.run(main) 