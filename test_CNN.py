import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from corpus_generator import *
from QRWCNN_arch import *
# Saving files
import os, inspect  # for current directory
from tensorflow.keras.models import Model
current_file_directory = os.path.dirname(os.path.abspath(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.config.list_physical_devices('GPU')

corpus_list=[np.array([[0., 1., 0., 0., 1., 0.],
               [1., 0., 0., 1., 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 1., 1., 0., 0., 0.],
               [1., 0., 0., 0., 0., 1.],
               [0., 0., 0., 0., 1., 0.]]),
        np.array([[0., 0., 1., 0., 0., 1.],
               [0., 0., 0., 0., 1., 0.],
               [1., 0., 0., 1., 0., 0.],
               [0., 0., 1., 0., 0., 0.],
               [0., 1., 0., 0., 0., 1.],
               [1., 0., 0., 0., 1., 0.]])]
label_list = [np.array([1., 0.]), np.array([0., 1.])]

n = 6
N = 2
data_X = np.ones((N, n, n))
data_labels = np.ones((N,2))
for i in range(N): 
    x = corpus_list[i]
    data_X[i] = x # numpy array
    data_labels[i] = label_list[i] # 2 dim np array, categorical

data_X = data_X.reshape(N, n, n, 1) # [samples, rows, columns, channels]
X_train = data_X[0].reshape(1, n, n, 1)
X_test = data_X[1].reshape(1, n, n, 1)
y_train = data_labels[0].reshape(1,2)
y_test = data_labels[1].reshape(1,2)


print(X_train.reshape((6,6)), y_train)
#print(X_test, y_test)

print('(N, n, n, C): ', X_train.shape)

print('\nTraining spec.:')
print('N_train: ', X_train.shape[0])
q_count_train = (y_train[:,1] == 1.0).sum()
print('q_percentage: %.2f' % (q_count_train/X_train.shape[0]))

print('\nTest spec.:')
print('N_test: ', X_test.shape[0])
q_count_test = (y_test[:,1] == 1.0).sum()
print('q_percentage: %.2f' %(q_count_test/X_test.shape[0]))

assert not np.any(np.isnan(y_test))
assert np.sum(y_test) == y_test.shape[0] # any ties  


num_classes = 2

y_upper = 1.0
#batch_size = 1
epochs = 1
batch_size = 1

model = ETE_ETV_Net(n, num_classes, conv_learn = True)
model = model.build( batch_size = batch_size)


data = (X_train, y_train)
'''
print(X_train.reshape(1,1,n,n))
for layer in range(len(model.layers)):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[layer].output)
    intermediate_output = intermediate_layer_model.predict(data)
    intermediate_layer_model.summary()
    print(intermediate_output)
    try:
        print(intermediate_output.reshape((1, 1, n,n)))
    except:
        pass
    
    input()
'''
print('predicted data ', model.predict(data))


history = model.fit(X_train, y_train, 
                        steps_per_epoch = 1,
                        batch_size=batch_size,
                        validation_data = (X_test, y_test),
                        epochs=epochs, verbose=2)
print('predicted data ', model.predict((X_test, y_test)))
'''
for layer in range(1, len(model.layers)):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[layer].output)
    intermediate_output = intermediate_layer_model.predict(data)
    intermediate_layer_model.summary()
    print(intermediate_output)
    try:
        print(intermediate_output.reshape((1,1, n,n)))
    except:
        pass
    
    input()
'''
'''
plt.figure(10)
plt.title('All')
plt.ylim(0.0, y_upper)
y = history.history['val_loss']
plt.plot(np.linspace(0.0, len(y), len(y)), y,'--', color = colors[i], label = 'test loss for ' + str(n) + ' nodes')
y = history.history['val_accuracy']
plt.plot(np.linspace(0.0, len(y), len(y)), y,'-', color = colors[i], label = 'test accuracy for ' + str(n) + ' nodes')
plt.xlabel('epochs')
plt.ylabel('learning performance')
plt.legend()
plt.savefig(current_file_directory + '/results_main' +'/All')

plt.figure(i)
plt.title(str(model_list[i]) + ' took time [min]: ' + str(round(vtime4/60,3)))
plt.ylim(0.0, y_upper)
y = history.history['val_loss']
plt.plot(np.linspace(0.0, len(y), len(y)), y,'--', color = tuple(t+0.1 for t in colors[i]), label = 'test loss')
y = history.history['loss']
plt.plot(np.linspace(0.0, len(y), len(y)), y,':', color = colors[i], label = 'loss')
y = history.history['val_accuracy']
plt.plot(np.linspace(0.0, len(y), len(y)), y,'-', color = tuple(t-0.1 for t in colors[i]), label = 'test accuracy')
y = history.history['accuracy']
plt.plot(np.linspace(0.0, len(y), len(y)), y,'-.', color = colors[i], label = 'accuracy')
plt.xlabel('epochs')
plt.ylabel('learning performance')
plt.legend()
plt.savefig(current_file_directory + '/results_main' +'/model ' + str(i))
np.savez(current_file_directory + '/results_main' +'/training_results' + str(i), history.history['loss'], history.history['val_accuracy'], history.history['val_loss'])

'''