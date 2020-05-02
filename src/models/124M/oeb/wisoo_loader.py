import tensorflow as tf
import tensorflow.contrib.slim as slim


with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('model.ckpt.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
  # print(type(new_saver))
  graph = tf.get_default_graph()

  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)