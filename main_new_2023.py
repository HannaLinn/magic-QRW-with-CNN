import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from corpus_generator import *
from QRWCNN_arch import *
from tensorflow.keras.utils import plot_model
from util_functions import *


# Saving files
import os, inspect  # for current directory
current_file_directory = os.path.dirname(os.path.abspath(__file__))
from sklearn.metrics import f1_score, precision_score, recall_score

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#tf.config.list_physical_devices('GPU')


batch_size = 10 #10
epochs = 50 #400
average_num = 10 #100

def add_ghost_nodes_dataset(data_X, num_nodes_test):
    '''
    Input: all matrices in a dataset.
    For each matrix, adds ghost nodes n+1, with no connection to the rest of the graph.
    Return: Dataset where all matrices are larger.
    '''
    n = data_X[0].shape[0]
    N = data_X.shape[0]
    data_X_nplus1 = np.ones((N, num_nodes_test, num_nodes_test))
    # for each matrix
    for i, A in enumerate(data_X):
        A = A.reshape(n,n)
        for j in range(num_nodes_test-n):
            n_new = A.shape[0]
            A = np.concatenate((A, np.zeros((n_new, 1))), axis = 1)
            A = np.concatenate((A, np.zeros((1, n_new+1))), axis = 0)
        data_X_nplus1[i] = A
    data_X_nplus1 = data_X_nplus1.reshape(N, num_nodes_test, num_nodes_test, 1)
    return data_X_nplus1

def main(num, now_testing, comp_list, net_type, num_nodes, generalisation, num_nodes_test):
    file_dir = current_file_directory + '/new_magic_results/' + now_testing
    names = ['c', 'q', 'positive', 'negative', 'T', 'H']
    
    num_classes = len(comp_list)
    names = [names[x] for x in comp_list]

    data_X, data_labels, data_ranking = load_data( current_file_directory + '/datasets/linear_graph_datasets/train_val_test_data_n_' + str(num_nodes) + '.npz')
    data_labels = choose_types(comp_list, data_ranking)

    print('Before rinse (N, n, n, C), labels: ', data_X.shape, data_labels.shape,)
    data_X, data_labels = delete_ties(data_X, data_labels)

    # make sure 50/50
    q_count_train = (data_labels[:,1] == 1.0).sum()
    print(names[1], '_percentage before: %.2f' % (q_count_train/data_labels.shape[0]))
    num_c=0
    num_q=0
    for i in data_labels:
        if i[0]==1:
            num_c=num_c+1
        else:
            num_q=num_q+1
    print('num_c=',num_c)
    print('num_q=',num_q)
    if num_c > num_q:
        adv_rray=[1., 0.]
    else:
        adv_rray=[0., 1.]
    data_X, data_labels = rinse_out(data_X, data_labels, adv_rray, abs(num_c-num_q))
    print('rinse: ', abs(num_c-num_q))

    print('\nData spec.:')
    print('N_train: ', data_X.shape[0])
    q_count_train = (data_labels[:,1] == 1.0).sum()
    print(names[1], '_percentage: %.2f' % (q_count_train/data_labels.shape[0]))

    if generalisation:
        print('\nGeneralisation:')
        data_X_test, data_labels_test, data_ranking_test = load_data(current_file_directory + '/datasets/linear_graph_datasets/train_val_test_data_n_' + str(num_nodes_test) + '.npz')
        data_labels_test = choose_types(comp_list, data_ranking_test)

        data_X_test, data_labels_test = delete_ties(data_X_test, data_labels_test)

        print('\nData spec.:')
        print('N_test: ', data_X_test.shape[0])
        q_count_train_test = (data_labels_test[:,1] == 1.0).sum()
        print(names[1], '_percentage: %.2f' % (q_count_train_test/data_labels_test.shape[0]))



    grande_train_loss = np.zeros((epochs, average_num))
    grande_train_accuracy = np.zeros((epochs, average_num))
    grande_test_accuracy = np.zeros((epochs, average_num))
    grande_test_loss = np.zeros((epochs, average_num))
    grande_precision = np.zeros((num_classes, average_num))
    grande_recall = np.zeros((num_classes, average_num))
    grande_f1 = np.zeros((num_classes, average_num))
    i = 0

    plt.figure(np.random.randint(1, 30))
    for average_iter in range(average_num):
        if not generalisation:
            X_train, X_test, y_train, y_test = train_test_split(data_X, data_labels, test_size=0.2)
        else:
            data_X = add_ghost_nodes_dataset(data_X, num_nodes_test=num_nodes_test)
            X_train, y_train = data_X, data_labels

            X_test, y_test = data_X_test, data_labels_test

        X_train, y_train = div_by_batch_size(X_train, y_train, batch_size)
        X_test, y_test = div_by_batch_size(X_test, y_test, batch_size)

        print('Train (N, n, n, C): ', X_train.shape)
        print('Test (N, n, n, C): ', X_test.shape)

        print('\nTrain spec.:')
        print('N_train: ', X_train.shape[0])
        q_count_train = (y_train[:,1] == 1.0).sum()
        print(names[1], '_percentage: %.2f' % (q_count_train/X_train.shape[0]))


        print('\nTest spec.:')
        print('N_test: ', X_test.shape[0])
        q_count_test = (y_test[:,1] == 1.0).sum()
        print(names[1],'_percentage: %.2f' %(q_count_test/X_test.shape[0]))

        assert not np.any(np.isnan(y_test))
        assert np.sum(y_test) == y_test.shape[0] # any ties

        y_upper = 1.0

        assert len(X_train) % batch_size == 0
        assert len(X_test) % batch_size == 0
        assert np.all(np.sum(y_train, axis = 1) == 1.) # no ties

        
        validation_freq = 1

        file2 = open(file_dir +'train_param.txt', 'w')
        q_count_train = (y_train[:,1] == 1.0).sum()
        q_count_test = (y_test[:,1] == 1.0).sum()
        L = ['batch_size: ' + str(batch_size) + '\n', 'epochs: ' + str(epochs) + '\n', 'train (N, n, n, C): ' + str(X_train.shape)+ ' q_percentage: %.2f' % (q_count_train/X_train.shape[0]) + '\n', 'test (N, n, n, C): ' + str(X_test.shape)+ ' q_percentage: %.2f' %(q_count_test/X_test.shape[0]) + '\n']
        file2.writelines(L)
        file2.close()

        X_train, y_train, X_test, y_test, = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test)

        print('-'*20, ' average iter: ', str(average_iter), ' n: ', str(num_nodes), '-'*20)

        #model_list = [[(False, True, 2, 10, 3), ((0.0, 0.0), 1000., 0.2)]]

        model = ETE_ETV_Net(num_nodes_test, num_classes = num_classes, net_type = net_type, conv_learn = True, num_ETE = 2, num_neurons = 10, depth_of_dense=3)
        model = model.build(batch_size = batch_size, reg_lambdas = (0.0, 0.0), con_norm = 1000., dropout_rate = 0.2)
        #plot_model(model, to_file=file_dir + 'model_plot' + str(i) + 'm.png', show_shapes=True, show_layer_names=True)
        start = time.time()
        callbacks = [tf.keras.callbacks.TerminateOnNaN()]
        history = model.fit(X_train, y_train, callbacks=callbacks, batch_size=batch_size, validation_data = (X_test, y_test), validation_freq = validation_freq, epochs=epochs, verbose=2, shuffle = True)        
        
        end = time.time()
        vtime4 = end-start

        y_pred1 = model.predict(X_test, batch_size=batch_size)
        y_pred = np.eye(1, num_classes, k=np.argmax(y_pred1, axis =1)[0])
        for pred in range(1, y_pred1.shape[0]):
            y_pred = np.append(y_pred, np.eye(1, num_classes, k=np.argmax(y_pred1, axis = 1)[pred]), axis = 0)

        grande_train_loss[:, average_iter] = history.history['loss']
        grande_test_accuracy[:, average_iter] = history.history['val_accuracy']
        grande_train_accuracy[:, average_iter] = history.history['accuracy']
        grande_test_loss[:, average_iter] = history.history['val_loss']
        grande_precision[:, average_iter] = precision_score(y_test, y_pred, average=None)
        grande_recall[:, average_iter] = recall_score(y_test, y_pred, average=None)
        grande_f1[:, average_iter] = f1_score(y_test, y_pred, average=None)

        np.savetxt(file_dir + 'grande_train_loss.out', grande_train_loss, delimiter=',')
        np.savetxt(file_dir + 'grande_test_accuracy.out', grande_test_accuracy, delimiter=',')
        np.savetxt(file_dir + 'grande_train_accuracy.out', grande_train_accuracy, delimiter=',')
        np.savetxt(file_dir + 'grande_test_loss.out', grande_test_loss, delimiter=',')
        np.savetxt(file_dir + 'grande_precision.out', grande_precision, delimiter=',')
        np.savetxt(file_dir + 'grande_recall.out', grande_recall, delimiter=',')
        np.savetxt(file_dir + 'grande_f1.out', grande_f1, delimiter=',')

        plt.title('took time [min]: ' + str(round(vtime4/60,3)) + str(names) + 'n: ' + str(num_nodes_test))
        plt.ylim(0.0, y_upper)
        y = history.history['val_loss']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'r--', label = 'test loss')
        y = history.history['loss']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'g:', label = 'train loss')
        y = history.history['val_accuracy']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'b-', label = 'test accuracy')
        y = history.history['accuracy']
        plt.plot(np.linspace(0.0, len(y), len(y)), y,'m-.', label = 'train accuracy')
        plt.xlabel('epochs', fontsize = 15)
        plt.ylabel('learning performance', fontsize = 15)
        if average_iter == 0:
            plt.legend()
        plt.savefig(file_dir + 'in_loop' + '.pdf')
        np.savez(file_dir + 'in_loop' + '.npz', history.history['loss'], history.history['val_accuracy'], history.history['val_loss'])
        
    
    train_loss = np.average(grande_train_loss, 1)
    test_accuracy = np.average(grande_test_accuracy, 1)
    train_accuracy = np.average(grande_train_accuracy, 1)
    test_loss = np.average(grande_test_loss, 1)
    precision = np.average(grande_precision, 1)
    recall = np.average(grande_recall, 1)
    f1 = np.average(grande_f1, 1)

    file1 = open(file_dir +'_f1_precision_recall_' + str(num_nodes_test) + '.txt', 'w')
    L = ['model ' + str(i) + '\n' + 
    'precision: ' + str(precision_score(y_test, y_pred, average=None))+ '\n' +
    'recall: ' + str(recall_score(y_test, y_pred, average=None)) +'\n' + 
    'f1: ' + str(f1_score(y_test, y_pred, average=None)) + '\n\n\n']
    file1.writelines(L)
    file1.close()

    plt.figure(np.random.randint(31, 40))
    plt.title('linear with types '  + str(names) + 'n: ' + str(num_nodes_test))
    plt.ylim(0.0, y_upper)
    plt.plot(np.linspace(0.0, len(train_loss), len(train_loss)), train_loss,':', label = 'train loss ' + str(round(train_loss[-1],2)))
    plt.plot(np.linspace(0.0, len(test_accuracy), len(test_accuracy)), test_accuracy,'-', label = 'test accuracy ' + str(round(test_accuracy[-1],2)))
    plt.plot(np.linspace(0.0, len(train_accuracy), len(train_accuracy)), train_accuracy,'-.', label = 'train accuracy ' + str(round(train_accuracy[-1],2)))
    plt.plot(np.linspace(0.0, len(test_loss), len(test_loss)), test_loss,'--', label = 'test loss ' + str(round(test_loss[-1],2)))
    plt.xlabel('epochs', fontsize = 15)
    plt.ylabel('learning performance', fontsize = 15)
    plt.legend()
    plt.grid()
    np.savez(file_dir + 'average_' + str(num_nodes_test) + '.npz', train_loss, test_accuracy, train_accuracy, test_loss)

    plt.savefig(file_dir + 'average_' + str(average_num) + '.pdf')



'''
for j in [6]:
    # ------------------------------------------------------------- 
    for i in [1,2,3]:
        
        name = 'num_nodes_' + str(j) + '_nettype_' + str(i) 

        now_testing = name + 'c_q_'
        print(now_testing)
        comp_list = [0, 1]
        main(num = i, now_testing = now_testing, comp_list = comp_list, net_type = i, num_nodes = j)

        # ------------------------------------------------------------- 
        now_testing = name + 'c_T_'
        print(now_testing)
        comp_list = [0, 4]
        main(num = i, now_testing = now_testing, comp_list = comp_list, net_type = i, num_nodes = j)

        # ------------------------------------------------------------- 
        now_testing = name + 'q_T_'
        print(now_testing)
        comp_list = [1, 4]
        main(num = i, now_testing = now_testing, comp_list = comp_list, net_type = i, num_nodes = j)
        
'''

# träna på 20 och 6, testa på alla andra
import time

start_time = time.time()

linear_cyclic = 'linear_'
test_j = 0

for j in [6, 20]:
    # time each loop
    start_time_loop = time.time()
    end_time_loop = time.time()
    elapsed_time_loop = end_time_loop - start_time_loop
    elapsed_minutes_loop = elapsed_time_loop / 60
    print(f"Elapsed time loop : {elapsed_minutes_loop:.2f} minutes")
    print("test_j: ", test_j)
    print("j: ", j)
    # ------------------------------------------------------------- 
    for arch in [1,2,3]:

        for test_j in range(j,j+5): # 6,7,8,9,10,   20,21,22,23,24,25
            print()
            name = 'num_nodes_' + str(j) + '_nettype_' + str(arch) + '_test_' + str(test_j) + '__'
            
            now_testing = name + 'c_q_' + linear_cyclic + '_'
            print(now_testing)
            comp_list = [0, 1]
            if j == test_j:
                main(num = arch, now_testing = now_testing, comp_list = comp_list, net_type = arch, num_nodes = j, generalisation = False, num_nodes_test = test_j)
            else:
                main(num = arch, now_testing = now_testing, comp_list = comp_list, net_type = arch, num_nodes = j, generalisation = True, num_nodes_test = test_j)

        if j == 20:
            test_j = 25

            name = 'num_nodes_' + str(j) + '_nettype_' + str(arch) + '_test_' + str(test_j)
            
            now_testing = name + 'c_q_'
            print(now_testing)
            main(num = arch, now_testing = now_testing, comp_list = comp_list, net_type = arch, num_nodes = j, generalisation = True, num_nodes_test = test_j)
            



# ------------------------------------------------
print('-'*20, ' DONE ', '-'*20)

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60
print(f"Elapsed time: {elapsed_minutes:.2f} minutes")