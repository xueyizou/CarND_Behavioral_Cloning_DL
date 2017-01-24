import numpy as np
from scipy.misc import imread
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Activation, Flatten, Lambda
from keras.optimizers import Adam
from keras.models import model_from_json

from preprocessor import csvfile_path, preprocessed_data_path, steer_offset, h, w

epochs = 15
batch_size = 128


def _read_imgs(img_names):
    imgs=[]
    for img_name in img_names:
        imgs.append(imread(preprocessed_data_path+img_name))
    return np.array(imgs)


def gen_train(img_names_train,steers_train, batch_size):
    """
    python generator to generate batch_size data for trainning. Note that I used sample_weights to offset the influence of
    the dominant data classes
    """
    def _f():
        start = 0
        end = start + batch_size
        n = len(img_names_train)

        while True:
            X_batch = _read_imgs(img_names_train[start:end])
            y_batch = steers_train[start:end]
            sample_weights = np.ones_like(y_batch)
            sample_weights[y_batch==0] = 0.5
            sample_weights[y_batch==steer_offset] = 0.5
            sample_weights[y_batch==-steer_offset] = 0.5

            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size

            yield (X_batch, y_batch, sample_weights)

    return _f


def gen_val(img_names_val,steers_val, batch_size):
    """
    python generator to generate batch_size data for validation.
    """
    def _f():
        start = 0
        end = start + batch_size
        n = len(img_names_val)

        while True:
            X_batch = _read_imgs(img_names_val[start:end])
            y_batch = steers_val[start:end]
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size

            yield (X_batch, y_batch)

    return _f



def create_model():
    """this model is based on paper titled "End to End Learning for Self-Driving Cars" paper
        url: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    activation = 'relu'

    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(h, w, 3)))

    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

#    model.add(Dropout(.5))

    model.add(Dense(1164))
    model.add(Activation(activation))

    model.add(Dense(100))
    model.add(Activation(activation))

    model.add(Dense(50))
    model.add(Activation(activation))

    model.add(Dense(10))
    model.add(Activation(activation))

    model.add(Dense(1))

    return model


def run_train():
    img_names_train=np.load('img_names_train.npy')
    img_names_val=np.load('img_names_val.npy')
    steers_train=np.load('steers_train.npy')
    steers_val=np.load('steers_val.npy')

    num_train = len(img_names_train)
    num_val = len(img_names_val)
    print("number of training data: ", num_train)
    print("number of validation data: ",num_val)

    model = create_model()
    model.summary()

    model.compile(optimizer=Adam(1e-4), loss="mse")

    # create two generators for training and validation
    train_gen = gen_train(img_names_train,steers_train, batch_size)
    val_gen = gen_val(img_names_val,steers_val, batch_size)
    # train model
    #fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=None, validation_data=None, nb_val_samples=None, class_weight=None, max_q_size=10, nb_worker=1, pickle_safe=False, initial_epoch=0)
    model.fit_generator(train_gen(), samples_per_epoch=num_train, nb_epoch=epochs, validation_data=val_gen(), nb_val_samples=num_val, verbose=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def run_prediction():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=Adam(1e-4), loss="mse" )

    import csv
    reader = csv.reader(open(csvfile_path, newline=''), delimiter=',')

    imgs = []
    steers=[]
    steers_predict=[]
    counter = 0
    reader.__next__()
    for row in reader:
        center_file_name = row[0].strip().replace('IMG/','')
        img = imread(preprocessed_data_path+center_file_name)
        imgs.append(img)
        steers.append(float(row[3]))
        counter+=1
        print(counter)
        if counter%batch_size ==0:
            steers_predict.extend(loaded_model.predict_on_batch(np.asarray(imgs)))
            imgs=[]
    steers_predict.extend(loaded_model.predict_on_batch(np.asarray(imgs)))
    np.save("prediction.npy", list(zip(steers, steers_predict)))

    mse=sum(np.subtract(np.asarray(steers), np.asarray(steers_predict).reshape(counter))**2)/counter
    print("mse: %.5f" % mse)



if __name__ == '__main__':
    mode = "train" #"prediction", "train"
    if mode == "train":
        run_train()
    else:
        run_prediction()
        global cmp
        cmp=np.load('prediction.npy')


