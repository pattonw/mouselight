import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    with tf.Session() as s:

        a = tf.Variable(1)
        b = tf.Variable(2)
        c = a + b

        g = tf.gradients(c, [a])

        s.run(a.initializer)
        s.run(b.initializer)
        r = s.run([g])
        print(r)

    with tf.Session() as s:

        mask = tf.constant([False, True, True, False], dtype=np.bool)
        embedding = tf.Variable([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]],
            dtype=np.float32)

        total_sum = tf.reduce_sum(embedding)
        gradient = tf.gradients(total_sum, embedding)

        filtered = tf.boolean_mask(embedding, mask)
        filtered_sum = tf.reduce_sum(filtered)
        filtered_gradient = tf.gradients(filtered_sum, embedding)

        s.run(embedding.initializer)

        print(s.run([total_sum, gradient]))
        print(s.run([filtered_sum, filtered_gradient]))
