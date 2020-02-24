import tensorflow as tf


def main():
    weights = tf.constant([[0.1, 0.2, 0.3, 0.1],
                           [0.7, 0.9, 0.1, 0.2]])
    embeding = tf.constant([[0.1, 0.2, 0.3, 0.1]])
    norm_w = tf.nn.l2_normalize(weights, axis=0)
    norm_e = tf.nn.l2_normalize(embeding, axis=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        w_result = sess.run(weights)
        e_result = sess.run(embeding)
        w_norm_result = sess.run(norm_w)
        e_norm_result = sess.run(norm_e)

        print(w_result)
        print(e_result)
        print(w_norm_result)
        print(e_norm_result)


if __name__ == '__main__':
    main()
