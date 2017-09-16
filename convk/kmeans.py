"""K-means optimizer.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
import keras.backend as K


class KMeansOptimizer(object):
    """K-Means Optimizer."""

    def __init__(self, model):
        """General K-means Optimizer.

        Assume a sequential model

        Parameters
        ----------
        """
        self.model = model

        # collect all relevant layers
        self.layer_idx = []
        for idx in range(len(model.layers)):
            if "kmeans" in model.layers[idx].name:
                self.layer_idx.append(idx)

    def perform_iteration(self, data_batch):
        """Updates parameters."""

        inputs = data_batch
        for idx in range(len(self.layer_idx)):
            # get layer
            if idx == 0:
                # first k means layer
                layer_input = self.model.layers[0].input
            else:
                # otherwise use layer input
                layer_input = self.model.layers[self.layer_idx[idx]].input
            # layer output
            layer_output = self.model.layers[self.layer_idx[idx]].output
            # prepare next input
            if idx != len(self.layer_idx)-1:
                next_input_idx = self.layer_idx[idx+1]-1
                next_input = self.model.layers[next_input_idx].output

            # forward function
            forward_fun = K.function([self.model.layers[idx].input,
                                      K.learning_phase()],
                                     [self.model.layers[idx].output])
            # backward pass

            loss, inputs = kmeans_fun(inputs, 0)

        pass
        # get all relevant layer from names

        # iterate layers by their index
        # build function for output

        # change parameters
