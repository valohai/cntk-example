import json
import argparse

import cntk
import numpy as np

print(f'CNTK Devices: {cntk.device.all_devices()}')


def train(FLAGS):
    """
    Original: https://cntk.ai/pythondocs/CNTK_101_LogisticRegression.html

    :param FLAGS: argparse values
    """
    input_dim = 2
    num_output_classes = 2
    np.random.seed(0)

    # Helper function to generate a random data sample
    def generate_random_data_sample(sample_size, feature_dim, num_classes):
        # Create synthetic data using NumPy.
        y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

        # Make sure that the data is separable
        x = (np.random.randn(sample_size, feature_dim) + 3) * (y + 1)

        # Specify the data type to match the input variable used
        # later in the tutorial (default type is double)
        x = x.astype(np.float32)

        # Convert class 0 into the vector "1 0 0",
        # class 1 into the vector "0 1 0", ...
        class_ind = [y == class_number for class_number in range(num_classes)]
        y = np.asarray(np.hstack(class_ind), dtype=np.float32)
        return x, y

    # Create the input variables denoting the features and the label data.
    # Note: the input does not need additional info on the number
    # of observations (samples) since CNTK creates only the network
    # topology first
    feature = cntk.input_variable(input_dim, np.float32)

    my_dict = {}

    def linear_layer(input_var, output_dim):
        _input_dim = input_var.shape[0]
        weight_param = cntk.parameter(shape=(_input_dim, output_dim))
        bias_param = cntk.parameter(shape=output_dim)
        my_dict['w'], my_dict['b'] = weight_param, bias_param
        return cntk.times(input_var, weight_param) + bias_param

    output_dim = num_output_classes
    z = linear_layer(feature, output_dim)  # output of the network

    label = cntk.input_variable(num_output_classes, np.float32)
    loss = cntk.cross_entropy_with_softmax(z, label)
    eval_error = cntk.classification_error(z, label)

    lr_schedule = cntk.learning_rate_schedule(FLAGS.learning_rate, cntk.UnitType.minibatch)
    learner = cntk.sgd(z.parameters, lr_schedule)
    trainer = cntk.Trainer(z, (loss, eval_error), [learner])

    minibatch_size = FLAGS.minibatch_size
    num_samples_to_train = FLAGS.sample_count
    num_minibatches_to_train = int(num_samples_to_train / minibatch_size)

    # Run the trainer and perform model training
    training_progress_output_freq = FLAGS.sample_count / 400

    for i in range(0, num_minibatches_to_train):
        features, labels = generate_random_data_sample(
            minibatch_size,
            input_dim,
            num_output_classes
        )

        # Assign the minibatch data to the input variables
        # and train the model on the minibatch
        trainer.train_minibatch({feature: features, label: labels})

        if i % training_progress_output_freq == 0:
            training_loss = trainer.previous_minibatch_loss_average
            eval_error = trainer.previous_minibatch_evaluation_average
            print(json.dumps({
                'minibatch': i,
                'loss': training_loss,
                'error': eval_error,
            }))

    # Run the trained model on a newly generated dataset
    test_minibatch_size = FLAGS.minibatch_size
    features, labels = generate_random_data_sample(test_minibatch_size, input_dim, num_output_classes)
    trainer.test_minibatch({feature: features, label: labels})

    out = cntk.softmax(z)
    result = out.eval({feature: features})

    print("Labels:", [np.argmax(label) for label in labels])
    print("Result:", [np.argmax(x) for x in result])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=25,
                        help='How many samples to work on for each iteration?')
    parser.add_argument('--sample_count', type=int, default=500000,
                        help='Train for how many observations?')
    parser.add_argument('--learning_rate', type=float, default=0.25,
                        help='Training learning rate')
    FLAGS, unparsed = parser.parse_known_args()
    train(FLAGS)
