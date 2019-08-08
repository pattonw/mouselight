from funlib.learn.tensorflow.models import unet, conv_pass
import tensorflow as tf
import json


def create_network(input_shape, name):
    tf.reset_default_graph()

    # c=2, d, h, w
    raw = tf.placeholder(tf.float32, shape=(2,) + input_shape)

    # b=1, c=2, d, h, w
    raw_batched = tf.reshape(raw, (1, 2) + input_shape)

    fg_unet = unet(raw_batched, 12, 5, [[1, 3, 3], [1, 2, 2], [2, 2, 2]])

    fg_batched = conv_pass(
        fg_unet[0], kernel_sizes=(1,), num_fmaps=1, activation="sigmoid"
    )

    output_shape_batched = fg_batched[0].get_shape().as_list()

    # d, h, w, strip the batch and channel dimension
    output_shape = tuple(output_shape_batched[2:])

    fg = tf.reshape(fg_batched[0], output_shape)

    labels_fg = tf.placeholder(tf.float32, shape=output_shape)
    loss_weights = tf.placeholder(tf.float32, shape=output_shape)

    loss = tf.losses.mean_squared_error(labels_fg, fg, loss_weights)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4, beta1=0.95, beta2=0.999, epsilon=1e-8
    )
    optimizer = opt.minimize(loss)

    print("input shape: %s" % (input_shape,))
    print("output shape: %s" % (output_shape,))

    tf.train.export_meta_graph(filename=name + ".meta")

    tf.train.export_meta_graph(filename="train_net.meta")

    names = {
        "raw": raw.name,
        "fg": fg.name,
        "loss_weights": loss_weights.name,
        "loss": loss.name,
        "optimizer": optimizer.name,
        "labels_fg": labels_fg.name,
    }

    with open(name + "_names.json", "w") as f:
        json.dump(names, f)

    config = {"input_shape": input_shape, "output_shape": output_shape, "out_dims": 1}
    with open(name + "_config.json", "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    create_network((80, 256, 256), "train_net")
