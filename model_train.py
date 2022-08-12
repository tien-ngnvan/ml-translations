import tensorflow as tf
import os
import time
from losses import compute_loss


def accuracy_function(real, pred):
    real = tf.cast(real, tf.int64)
    pred = tf.cast(pred, tf.int64)
    
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    
    mask = tf.math.logical_not(tf.math.equal(real, 1)) # padding = 0 
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

class SequenceTransducerTrainer(object):
    """Trains a SequenceTransducer model."""
    def __init__(self, model, label_smoothing):
        """Constructor.
        Args:
          model: an instance of sequence transducer model.
          label_smoothing: float scalar, applies label smoothing to the one-hot 
            class labels. Positive class has prob mass 1 - `label_smoothing`, while 
            each negative class has prob mass `label_smoothing / num_neg_classes`.
        """
        self._model = model
        self._label_smoothing = label_smoothing
        
    def train(self,
              train_set,
              val_set,
              epochs,
              optimizer,
              ckpt,
              ckpt_path,
              num_iterations,
              persist_per_iterations,
              clip_norm=1.0,
              log_per_iterations=500,
              logdir='log',):
        """Performs training iterations.
        Args:
          dataset: a tf.data.Dataset instance, the input data generator.
          optimizer: a tf.keras.optimizer.Optimizer instance, applies gradient 
            updates.
          ckpt: a tf.train.Checkpoint instance, saves or load weights to/from 
            checkpoint file.
          ckpt_path: string scalar, the path to the directory that the checkpoint 
            files will be written to or loaded from.
          num_iterations: int scalar, num of iterations to train the model.
          persist_per_iterations: int scalar, saves weights to checkpoint files
            every `persist_per_iterations` iterations.
          clip_norm: float scalar, the value that the norm of gradient will be 
            clipped to.
          log_per_iterations: int scalar, prints log info every `log_per_iterations`
            iterations.
          logdir: string scalar, the directory that the tensorboard log data will
            be written to. 
        """ 
        train_accuracy = tf.keras.metrics.Mean(name='train_acc')
        val_accuracy = tf.keras.metrics.Mean(name='val_acc')

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')

        ############################################################################################
    
        train_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

        @tf.function(input_signature=train_step_signature)
        def train_step(inputs, target):
            """Performs a single training step on a minibatch of source and target
            token ids.
            Args:
            src_token_ids: int tensor of shape [batch_size, src_seq_len], lists of
              subtoken ids of batched source sequences ending with EOS_ID and 
              zero-padded.
            tgt_token_ids: int tensor of shape [batch_size, src_seq_len], lists of
              subtoken ids of batched target sequences ending with EOS_ID and 
              zero-padded.
            Returns:
            loss: float scalar tensor, the loss.
            step: int scalar tensor, the global step.
            lr: float scalar tensor, the learning rate.
            """

            with tf.GradientTape() as tape:
                # for each sequence of subtokens s1, s2, ..., sn, 1
                # prepend it with 0 (SOS_ID) and truncate it to the same length:
                # 0, s1, s2, ..., sn
                #tgt_token_ids_input = tf.pad(tgt_token_ids, [[0, 0], [1, 0]])[:, :-1]
                tar_inp = target[:, :-1]
                tar_real = target[:, 1:]

                logits = self._model(inputs, tar_inp, training=True)
                loss = compute_loss(tar_real,
                                    logits,
                                    self._label_smoothing,
                                    self._model._vocab_size_tgt,
                                    padding_value=1)

            gradients = tape.gradient(loss, self._model.trainable_variables)
            if clip_norm is not None:
                gradients, norm = tf.clip_by_global_norm(gradients, clip_norm)
            optimizer.apply_gradients(
              zip(gradients, self._model.trainable_variables))

            step = optimizer.iterations
            #lr = optimizer.learning_rate(step)
            loss = train_loss(loss)
            acc = train_accuracy(accuracy_function(tar_real, logits))
            
            #return loss, step - 1, lr, acc
            return loss, acc, step-1
        
        #####################################################################################
        val_step_signature = [
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

        @tf.function(input_signature=val_step_signature)
        def val_step(inputs, target):
            """Performs a single training step on a minibatch of source and target
            token ids.
            Args:
            src_token_ids: int tensor of shape [batch_size, src_seq_len], lists of
              subtoken ids of batched source sequences ending with EOS_ID and 
              zero-padded.
            tgt_token_ids: int tensor of shape [batch_size, src_seq_len], lists of
              subtoken ids of batched target sequences ending with EOS_ID and 
              zero-padded.
            Returns:
            loss: float scalar tensor, the loss.
            step: int scalar tensor, the global step.
            lr: float scalar tensor, the learning rate.
            """            
            with tf.GradientTape() as tape:
                tar_inp = target[:, :-1]
                tar_real = target[:, 1:]

                logits = self._model(inputs, tar_inp, training=False)
                loss = compute_loss(tar_real,
                                    logits,
                                    self._label_smoothing,
                                    self._model._vocab_size_tgt,
                                    padding_value=1)
            loss = val_loss(loss)
            acc = val_accuracy(accuracy_function(tar_real, logits))
            
            return loss, acc

        ###################################### TRAINING ######################################
        summary_writer = tf.summary.create_file_writer(logdir)

        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=3)
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        else:
            print('Training from scratch...')
            
        step = 0
        while step < num_iterations:
            start = time.time()
            # train
            for (_, (src_token_ids, tgt_token_ids)) in enumerate(train_set):
                train_lss, train_acc, step = train_step(src_token_ids, tgt_token_ids)
                #total += step
                
                train_acc = tf.make_tensor_proto(train_acc)
                train_acc = tf.make_ndarray(train_acc)

                if step % log_per_iterations == 0:
                    print('Step: %d, train loss: %f, train acc:' % 
                            (step.numpy(), train_lss), train_acc)

                with summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_lss, step=step)
                    tf.summary.scalar('train_accuracy', train_acc, step=step)

                # save checkpoint
                if step % persist_per_iterations  == 0:
                    print('Saving checkpoint at global step %d ...' % step)
                    ckpt_save_path = ckpt_manager.save()
                    #ckpt.save(os.path.join(ckpt_path, 'transformer'))

            # validation set
            for (_, (src_token_val, tgt_token_val)) in enumerate(val_set):
                val_lss, val_acc = val_step(src_token_ids, tgt_token_ids)
            
            val_acc = tf.make_tensor_proto(val_acc)
            val_acc = tf.make_ndarray(val_acc)
            with summary_writer.as_default():
                tf.summary.scalar('val_loss', val_lss, step=step)
                tf.summary.scalar('val_accuracy', val_acc, step=step)
            print("\n---------------- Summary 1 epochs ----------------")
            print('----Global step: %d, train loss: %f, train acc:' %  (step, train_lss), train_acc)
            print('----Global step: %d, val loss: %f, val acc:' % (step, val_lss), val_acc)
            print(f'----Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
            print("--------------------------------------------------\n")
            
            
           
            