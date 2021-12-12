import sys
import numpy as np

def split_data(data, lines_of_train):
    train_set = data[ :lines_of_train]
    validation_set = data[lines_of_train: ]
    return train_set, validation_set


def zscore_normalization(data_set):
    return (data_set - data_set.mean(0)) / data_set.std(0)


class NN:
    def __init__(self):
        self.layers_size = []
        self.best_params = {}
        self.current_accuracy = 0
        self.learning_rate = 0.1

    def create_layer(self, num_of_neurons):
        self.layers_size.append(num_of_neurons)
    
    def weights_and_bias_init(self):
        layers_size = self.layers_size
        W1 = np.random.randn(layers_size[1], layers_size[0]) * 0.01
        b1 = np.zeros((layers_size[1], 1))
        W2 = np.random.randn(layers_size[2], layers_size[1]) * 0.01
        b2 = np.zeros((layers_size[2], 1))
        self.best_params =  {'W1' : W1, 'b1': b1, 'W2': W2, 'b2' :b2}
        
    def feed_forward(self, x, params):
        W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
        z1 = np.dot(W1, x) + b1
        h1 = self.sigmoid(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = self.softmax(z2)
        predicted_result = np.argmax(h2)
        return {'x':x, 'z1':z1, 'h1':h1, 'z2':z2, 'h2':h2, 'predicted_result': predicted_result}

    def back_prop(self, cache_from_forward, target, params):
        y = np.zeros((10,1))
        y[int(target)] = 1
        x = cache_from_forward['x']
        h2 = cache_from_forward['h2']
        z2 = cache_from_forward['z2']
        h1 = cache_from_forward['h1']
        z1 = cache_from_forward['z1']

        dz2 = h2 - y
        dW2 = np.dot(dz2, h1.T)
        db2 = dz2
        dz1 = np.dot(params['W2'].T, dz2) * self.sigmoid_derivative(z1)
        dW1 = np.dot(dz1, x.T)
        db1 = dz1

        W1 = params['W1'] - self.learning_rate * dW1
        b1 = params['b1'] - self.learning_rate * db1
        W2 = params['W2'] - self.learning_rate * dW2
        b2 = params['b2'] - self.learning_rate * db2
        return {'W1' : W1, 'b1': b1, 'W2': W2, 'b2' :b2}        
    
    def validate(self, validation_set, params):
        counter = 0
        for input_line, target in validation_set:
            input_line = input_line.reshape(len(input_line), 1)
            cache_from_forward = self.feed_forward(input_line, params)
            predicted_result = cache_from_forward['predicted_result']
            if predicted_result == int(target):
                counter += 1
        accuracy = (counter / len(validation_set)) * 100
        return accuracy

    def train(self, train_set, validation_set, num_of_epochs):
        self.weights_and_bias_init()
        for i in range(num_of_epochs):
            params = self.best_params.copy()
            np.random.shuffle(train_set)
            for input_line, target in train_set:
                input_line = input_line.reshape(len(input_line), 1)
                cache_from_forward = self.feed_forward(input_line, params).copy()
                params = self.back_prop(cache_from_forward, target, params).copy()
            accuracy = self.validate(validation_set, params)
            print(accuracy)
            if accuracy >= self.current_accuracy:
                self.current_accuracy = accuracy
                self.best_params = params.copy()
        return params

    def predict(self, test_set, params):
        test_y = open("test_y", 'w')
        for input_line in test_set:
            input_line = input_line.reshape(len(input_line), 1)
            cache_from_forward = self.feed_forward(input_line, params)
            predicted_result = cache_from_forward['predicted_result']
            test_y.write(str(predicted_result) + '\n')

    def softmax(self, x):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def Relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


if __name__ == "__main__":
    train_data = np.loadtxt(sys.argv[1])
    targets = np.loadtxt(sys.argv[2])
    test_data = np.loadtxt(sys.argv[3])
    test_targets = np.loadtxt("real_data/test_y")

    train_data = train_data / 255 # normalization
    test_data = test_data / 255
    
    union_train_data = list(zip(train_data,  targets))
    np.random.shuffle(union_train_data)

    union_test_data = list(zip(test_data, test_targets))

    # train_set ,validation_set = split_data(union_train_data, 49500)
    train_set = union_train_data
    test_set = union_test_data

    NN = NN()
    NN.create_layer(len(train_data[0]))   # input
    NN.create_layer(128)                  # hidden
    NN.create_layer(10)                   # output
    
    params = NN.train(train_set, test_set, 14)
    # NN.predict(test_data, params)
 








