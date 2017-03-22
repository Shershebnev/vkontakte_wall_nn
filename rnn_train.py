import math
import numpy as np
import time

import tensorflow as tf
from tensorflow.contrib import layers
# apparently will be moved back to code in TF 1.1
from tensorflow.contrib import rnn

from wall_posts_downloader import *
import txt_utils as txt

group_id = 74479926
output_dir = "data"
alphabet = downloader(group_id, output_dir, 100)

SEQLEN = 30
BATCHSIZE = 100
ALPHABET_SIZE = len(alphabet)
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate for now
dropout_pkeep = 0.75  # no dropout for now

filedir = "data/*.txt"  # FIXME: hardcoded
encoded_text, validation_text, bookranges = txt.read_data_files(filedir, validation = True)

# model
lr = tf.placeholder(tf.float32, name="lr")
pkeep = tf.placeholder(tf.float32, name="pkeep")
batchsize = tf.placeholder(tf.int32, name="batchsize")

# inputs
X = tf.placeholder(tf.uint8, [None, None], name="X")
Xo = tf.one_hot(X, ALPHABET_SIZE, 1.0, 0.0)
# outputs
Y_ = tf.placeholder(tf.uint8, [None, None], name="Y_")
Yo_ = tf.one_hot(Y_, ALPHABET_SIZE, 1.0, 0.0)
# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE * NLAYERS], name="Hin")

single_cell = rnn.GRUCell(INTERNALSIZE)
drop_cell = rnn.DropoutWrapper(single_cell, input_keep_prob=pkeep)
multiple_cell = rnn.MultiRNNCell([drop_cell] * NLAYERS, state_is_tuple=False)
multiple_cell = rnn.DropoutWrapper(multiple_cell, output_keep_prob=pkeep)
Yr, H = tf.nn.dynamic_rnn(multiple_cell, Xo, dtype=tf.float32, initial_state=Hin)

H = tf.identity(H, name="H")

# softmax layer
Yflatten = tf.reshape(Yr, [-1, INTERNALSIZE])
Ylogits = layers.linear(Yflatten, ALPHABET_SIZE)
Yflatten_ = tf.reshape(Yo_, [-1, ALPHABET_SIZE])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflatten_)
loss = tf.reshape(loss, [batchsize, -1])
Yo = tf.nn.softmax(Ylogits, name="Yo")
Y = tf.argmax(Yo, 1)
Y = tf.reshape(Y, [batchsize, -1], name="Y")
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# stats
seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# checkpoints
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)

# init
# initial zero input state
istate = np.zeros([BATCHSIZE, INTERNALSIZE * NLAYERS])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
current_epoch = -1
for x, y_, epoch in txt.rnn_minibatch_sequencer(encoded_text, BATCHSIZE, SEQLEN, 50):
    if epoch != current_epoch:
        print("\nSTARTING {} EPOCH\n".format(epoch))
        current_epoch = epoch
    feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate,
                 pkeep: dropout_pkeep, batchsize: BATCHSIZE}
    _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

    if step % _50_BATCHES == 0 and len(validation_text) > 0:
        print("DOING VALIDATION STEP")
        validation_seqlen = 2 * INTERNALSIZE
        bsize = len(validation_text) // validation_seqlen
        validation_x, validation_y, _ = next(txt.rnn_minibatch_sequencer(validation_text, bsize, validation_seqlen, 1))
        validation_nullstate = np.zeros([bsize, INTERNALSIZE * NLAYERS])
        feed_dict = {X: validation_x, Y_: validation_y,
                     Hin: validation_nullstate, pkeep: 1.0,
                     batchsize: bsize}
        val_loss, val_accuracy, val_smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        print("VALIDATION RESULTS: loss = {:.4f}, accuracy = {:.4f}".format(val_loss, val_accuracy))

    if step // 3 % _50_BATCHES == 0:
        print("DOING GENERATION STEP")
        ry = np.array([[txt.convert_from_alphabet(ord("В"))]])
        rh = np.zeros([1, INTERNALSIZE * NLAYERS])
        for k in range(570):  # 2 times average post length is ~235
            ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh,
                batchsize: 1})
            rc = txt.sample_text(ryo, 10 if epoch <= 1 else 2, ALPHABET_SIZE)
            print(chr(txt.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])
        print("\nFINISHED GENERATION STEP")

    if step // 10 % _50_BATCHES == 0:
        saver.save(sess, "checkpoints/rnn_train_"  + str(math.trunc(time.time())), global_step = step)
    istate = ostate
    step += BATCHSIZE * SEQLEN




