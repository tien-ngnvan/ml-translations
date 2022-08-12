import tensorflow as tf
import numpy as np

from transformer.Transformer_Model import TransformerModel


EOS_ID = 18990 # start of sentence 
SOS_ID = 54400 # end of sentence

class SequenceTransducerEvaluator(object):
    """Evaluates a sequence transducer model."""
    def __init__(self, model, tokenizer):
        """Constructor.
        Args:
          model: an instance of sequence transducer model. 
        """
        self._model = model
        self._tokenizer = tokenizer
        
    def translate(self, source_text, output_filename=None):
        """Translates the source sequences.
        Args:
          source_text: input is a list source_ids, target_ids each of 
              them has shape [batch, token_ids]\n'.
          output_filename: (Optional) string scalar, name of the file that 
              translations will be written to.
        Returns:
          translations: a list of strings, the translated sequences.
          sorted_indices: a list of ints, used to reorder the translated sequences.
        """
        translations = []
        for step, (source_ids, target_ids) in enumerate(source_text):
            # transduces `source_ids` into `translated_ids`, trims token ids at and 
            # beyond EOS, and decode token ids back to text
            translated_ids = self._model.transduce(source_ids)
            translated_ids = np.array(translated_ids)
            length = translated_ids.shape[0]
            for j in range(length):
                print(j)
                print(translated_ids[0].shape)
                break
                # loop in batch [32, 128]
                for idx in range(translated_ids[j].shape[0]):
                    translation = self._trim_and_decode(translated_ids[j][idx])
            #    translations.append(translation)

        # optionally write translations to a text file
        #if output_filename is not None:
        #    _write_translations(output_filename, sorted_indices, translations)
        return translations

    def evaluate(self,
                 source_text_filename, 
                 target_text_filename, 
                 output_filename=None):
        """Translates the source sequences and computes the BLEU score of the 
        translations against groundtruth target sequences.
        Args:
          source_text_filename: string scalar, name of the text file storing source 
            sequences, lines separated by '\n'.
          target_text_filename: (Optional) string scalar, name of the text file 
            storing target sequences, lines separated by '\n'.
          output_filename: (Optional) name of the file that translations will be 
            written to.
        Returns:
          case_insensitive_score: float scalar, BLEU score when all chars are 
            lowercased.
          case_sensitive_score: float scalar, BLEU score when all chars are in 
            original case.
        """
        translations, sorted_indices = self.translate(
            source_text_filename, output_filename)
        
        targets = tf.io.gfile.GFile(
            target_text_filename).read().strip().splitlines()

        # reorder translations to their original positions in the input file
        translations = [translations[i] for i in sorted_indices]

        # compute BLEU score if case-sensitive
        targets_tokens = [self._bleu_tokenizer.tokenize(x) for x in targets]
        translations_tokens = [self._bleu_tokenizer.tokenize(x) 
                               for x in translations]
        case_sensitive_score = corpus_bleu(
            [[s] for s in targets_tokens], translations_tokens) * 100

        # compute BLEU score if case-insensitive (lower case)
        targets = [x.lower() for x in targets]
        translations = [x.lower() for x in translations]
        targets_tokens = [self._bleu_tokenizer.tokenize(x) for x in targets]
        translations_tokens = [self._bleu_tokenizer.tokenize(x) 
                               for x in translations]
        case_insensitive_score = corpus_bleu(
            [[s] for s in targets_tokens], translations_tokens) * 100

        return case_insensitive_score, case_sensitive_score
    


    def _trim_and_decode(self, ids):
        """Trims tokens at EOS and beyond EOS in ids, and decodes the remaining ids 
        back to text.
        Args:
          ids: numpy array of shape [num_ids], the translated ids ending with EOS id
            and zero-padded. 
        Returns:
          string scalar, the decoded text string.
        """
        try:
            eos_index = list(ids).index(EOS_ID)
            ids2token = self._tokenizer.convert_ids_to_tokens(ids[:eos_index])
            print(self._tokenizer.convert_tokens_to_string(ids2token))
            return self._tokenizer.convert_tokens_to_string(ids2token)
           
        except ValueError:  # No EOS found in sequence
            ids2token = self._tokenizer.convert_ids_to_tokens(ids)
            return self._tokenizer.convert_tokens_to_string(ids2token)
    
def _decode(ids, tokenizer):
    for key, value in enumerate(tokenizer.get_vocab()):
        if key == ids: return value
        
def _write_translations(output_filename, sorted_indices, translations):
    """Writes translations to a text file.
    Args:
        output_filename: string scalar, name of the file that translations will be 
          written to.
        sorted_indices: a list of ints, `[translations[i] for i in sorted_indices]`
          would produce the translations of text lines in the original text file.
        translations: a list of strings, the tranlations. 
    """
    with tf.io.gfile.GFile(output_filename, "w") as f:
        for i in sorted_indices:
            f.write("%s\n" % translations[i])