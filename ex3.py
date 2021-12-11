import sys
import numpy as np

    
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
        b1 = np.random.randn(layers_size[1], 1) * 0.01
        W2 = np.random.randn(layers_size[2], layers_size[1]) * 0.01
        b2 = np.random.randn(layers_size[2], 1) * 0.01
        self.best_params =  {'W1' : W1, 'b1': b1, 'W2': W2, 'b2' :b2}
        
    def feed_forward(self, x, target, params):
        W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
        is_correct_prediction = False
        z1 = np.dot(W1, x) + b1
        h1 = self.sigmoid(z1)
        z2 = np.dot(W2, h1) + b2
        h2 = self.softmax(z2)
        predicted_result = np.where(h2 == h2.max())[0][0] + 1
        if predicted_result == target:
            is_correct_prediction = True
        return {'x':x, 'z1':z1, 'h1':h1, 'z2':z2, 'h2':h2, 'is_correct_prediction': is_correct_prediction}

    def back_prop(self, cache_from_forward, target, params):
        y = np.zeros(10)
        y[int(target) - 1] = 1
        y = y.reshape(len(y), 1)
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
            cache_from_forward = self.feed_forward(input_line, target, params)
            is_correct_result = cache_from_forward['is_correct_prediction']
            if is_correct_result == True:
                counter += 1
        accuracy = (counter / len(validation_set)) * 100
        return accuracy

    def train(self, train_set, validation_set, num_of_epochs):
        self.weights_and_bias_init()
        params = self.best_params
        for i in range(num_of_epochs):
            for input_line, target in train_set:
                input_line = input_line.reshape(len(input_line), 1)
                cache_from_forward = self.feed_forward(input_line, target, params)
                params = self.back_prop(cache_from_forward, target, params)
            accuracy = self.validate(validation_set, params)
            print(accuracy)
            if accuracy >= self.current_accuracy:
                self.current_accuracy = accuracy
                self.best_params = params

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

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
 
    train_data = train_data / 255 # normalization

    train_set = train_data[: 2000]
    validation_set = train_data[2000: ]

    targets_for_train = targets[: 2000]
    targets_for_validation = targets[2000: ]

    union_train_set = list(zip(train_set,  targets_for_train))
    np.random.shuffle(union_train_set)

    union_validation_set = list(zip(validation_set,  targets_for_validation))
    np.random.shuffle(union_validation_set)

    NN = NN()
    NN.create_layer(len(train_data[0]))   # input
    NN.create_layer(100)                  # hidden
    NN.create_layer(10)                   # output
    
    NN.train(union_train_set, union_validation_set, 5)
 








