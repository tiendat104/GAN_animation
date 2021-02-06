import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from numpy.random import choice

import warnings
warnings.filterwarnings('ignore')

import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, ReLU ,BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Conv2DTranspose
import tensorflow as tf

UNROLLED_STEPS = 10
def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)

def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath


def load_images(directory='', size=(64, 64)):
    images = []
    labels = []  # Integers corresponding to the categories in alphabetical order
    label = 0

    imagePaths = list(list_images(directory))

    for path in imagePaths:

        if not ('OSX' in path):
            path = path.replace('\\', '/')

            image = cv2.imread(path)  # Reading the image with OpenCV
            image = cv2.resize(image, size)  # Resizing the image, in case some are not of the same size

            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return images

images=load_images('data')

def noise_label(y, p_flip):
    n_select = int(p_flip * y.shape[0])
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    y[flip_ix] = 1 - y[flip_ix]
    return y

class GAN():
    def __init__(self):
        self.img_shape = (64, 64, 3)

        self.noise_size = 200

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        loss_D = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)
        self.discriminator.compile(loss=loss_D,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = self.define_gan(self.generator, self.discriminator)

    def build_generator(self):
        epsilon = 0.00001  # Small float added to variance to avoid dividing by zero in the BatchNorm layers.
        noise_shape = (self.noise_size,)

        model = Sequential()

        model.add(Dense(4 * 4 * 256, activation='linear', input_shape=noise_shape))
        model.add(Reshape((4, 4, 256)))

        model.add(Conv2DTranspose(256, kernel_size=[2, 2], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=[2, 2], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(128, kernel_size=[2, 2], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2DTranspose(64, kernel_size=[2, 2], strides=[2, 2], padding="same",
                                  kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)))
        model.add(BatchNormalization(momentum=0.9, epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(3, kernel_size=[3, 3], strides=[1, 1], padding="same"))

        # Standard activation for the generator of a GAN
        model.add(Activation("tanh"))

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), strides=2, padding='same', input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def define_gan(self, g_model, d_model):
        d_model.trainable = False
        gen_noise = g_model.input
        gen_output = g_model.output
        gan_output = d_model(gen_output)
        model = Model(gen_noise, gan_output)

        opt = Adam(0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        d_model.trainable = True
        return model

    def train(self, epochs, batch_size=128, save_images_interval=100, save_model_interval=2000, interval_history = 100):
        os.makedirs("save", exist_ok=True)
        subs = os.listdir("save")
        new_subdir = os.path.join("save", str(len(subs) + 1))
        os.makedirs(new_subdir, exist_ok=False)
        new_img_dir = os.path.join(new_subdir, "image")
        new_weight_dir = os.path.join(new_subdir, "weight")
        os.makedirs(new_img_dir)
        os.makedirs(new_weight_dir)

        X_train = np.array(images)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = int(batch_size / 2)
        dict_history = {"D_loss_real": [], "D_loss_fake": [], "D_acc_real": [], "D_acc_fake": [], "G_loss":[]}
        tmp_dict_history = {"D_loss_real": [], "D_loss_fake": [], "D_acc_real": [], "D_acc_fake": [], "G_loss":[]}

        for epoch in range(1, epochs+1):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise_size))
            gen_imgs = self.generator.predict(noise)

            # Training the discriminator

            # The loss of the discriminator is the mean of the losses while training on authentic and fake images
            true_labels = np.ones((half_batch, 1))
            noise_true_labels = noise_label(true_labels, 0.05)
            d_real_statics = self.discriminator.train_on_batch(imgs, noise_true_labels)
            d_loss_real = d_real_statics[0]
            d_acc_real = d_real_statics[1]

            fake_labels = np.zeros((half_batch, 1))
            noise_fake_labels = noise_label(fake_labels, 0.05)
            d_fake_statics = self.discriminator.train_on_batch(gen_imgs, noise_fake_labels)
            d_loss_fake = d_fake_statics[0]
            d_acc_fake = d_fake_statics[1]

            tmp_dict_history["D_loss_real"].append(d_loss_real)
            tmp_dict_history["D_loss_fake"].append(d_loss_fake)
            tmp_dict_history["D_acc_real"].append(d_acc_real)
            tmp_dict_history["D_acc_fake"].append(d_acc_fake)

            # Training the generator
            backup_D = self.build_discriminator()
            weights_discriminator = self.discriminator.get_weights()
            backup_D.set_weights(weights_discriminator)

            for _ in range(UNROLLED_STEPS):
                tmp_idx = np.random.randint(0, X_train.shape[0], half_batch)
                tmp_imgs = X_train[tmp_idx]
                tmp_noise = np.random.normal(0,1, (half_batch, self.noise_size))
                tmp_gen_imgs = self.generator.predict(tmp_noise)

                tmp_true_labels = np.ones((half_batch,1))
                tmp_noise_true_labels = noise_label(tmp_true_labels, 0.05)
                tmp_d_real_statics = self.discriminator.train_on_batch(tmp_imgs, tmp_noise_true_labels)

                tmp_fake_labels = np.zeros((half_batch, 1))
                tmp_noise_fake_labels = noise_label(tmp_fake_labels, 0.05)
                tmp_d_fake_statics = self.discriminator.train_on_batch(tmp_gen_imgs, tmp_noise_fake_labels)

            noise = np.random.normal(0, 1, (batch_size, self.noise_size))
            valid_y = np.array([1] * batch_size)
            noise_valid_y = noise_label(valid_y, 0.05)
            g_loss = self.combined.train_on_batch(noise, noise_valid_y)
            tmp_dict_history["G_loss"].append(g_loss)

            # unroll discriminator
            backup_D_weights = backup_D.get_weights()
            self.discriminator.set_weights(backup_D_weights)

            # save history
            if epoch % interval_history == 0:
                dict_history["D_loss_real"].append(sum(tmp_dict_history["D_loss_real"])/len(tmp_dict_history["D_loss_real"]))
                dict_history["D_loss_fake"].append(sum(tmp_dict_history["D_loss_fake"])/len(tmp_dict_history["D_loss_fake"]))
                dict_history["D_acc_real"].append(sum(tmp_dict_history["D_acc_real"]) / len(tmp_dict_history["D_acc_real"]))
                dict_history["D_acc_fake"].append(sum(tmp_dict_history["D_acc_fake"]) / len(tmp_dict_history["D_acc_fake"]))
                dict_history["G_loss"].append(sum(tmp_dict_history["G_loss"]) / len(tmp_dict_history["G_loss"]))
                tmp_dict_history["D_loss_real"].clear()
                tmp_dict_history["D_loss_fake"].clear()
                tmp_dict_history["D_acc_real"].clear()
                tmp_dict_history["D_acc_fake"].clear()
                tmp_dict_history["G_loss"].clear()
                print("epoch ", epoch, ": D_loss_real = ", dict_history["D_loss_real"][-1], ", D_loss_fake = ",
                      dict_history["D_loss_fake"][-1], ", D_acc_real = ", dict_history["D_acc_real"][-1],
                      ", D_acc_fake = ", dict_history["D_acc_fake"][-1], ", G_loss = ", dict_history["G_loss"][-1])
                self.plot_history(dict_history)

            # Saving 25 images
            if epoch % save_images_interval == 0:
                self.save_images(epoch, new_img_dir)

            # We save the architecture of the model, the weights and the state of the optimizer
            # This way we can restart the training exactly where we stopped
            if epoch % save_model_interval == 0:
                self.save_model(self.discriminator, self.generator, self.combined, save_dir=new_weight_dir, epoch=epoch)

    def save_model(self, discriminator, generator, gan, save_dir, epoch):
        d_path = os.path.join(save_dir, "discriminator_epoch_" + str(epoch) + ".h5")
        g_path = os.path.join(save_dir, "generator_epoch_" + str(epoch) + ".h5")
        gan_path = os.path.join(save_dir, "gan_epoch_" + str(epoch) + ".h5")

        discriminator.trainable = False
        gan.save(gan_path)
        discriminator.trainable = True
        generator.save(g_path)
        discriminator.save(d_path)

    def save_images(self, epoch, save_dir):
        side = 10
        noise = np.random.normal(0, 1, (side*side, self.noise_size))
        gen_imgs = self.generator.predict(noise)

        # Rescale from [-1,1] into [0,1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(side, side, figsize=(16, 16))

        for i in range(side):
            for j in range(side):
                axs[i, j].imshow(gen_imgs[side * i + j])
                axs[i, j].axis('off')

        plt.show()
        save_path = os.path.join(save_dir, "Faces_%d.png" % epoch)
        fig.savefig(save_path)
        plt.close()

    def plot_history(self, dict_):
        # dict_history = {"D_loss_real": [], "D_loss_fake": [], "D_acc_real": [], "D_acc_fake": [], "G_loss":[]}
        d_loss_real = dict_["D_loss_real"]
        d_acc_real = dict_["D_acc_real"]
        d_loss_fake = dict_["D_loss_fake"]
        d_acc_fake = dict_["D_acc_fake"]
        g_loss = dict_["G_loss"]

        fig = plt.figure(figsize=(30, 10))
        fig.add_subplot(1, 3, 1)
        plt.plot(d_loss_real, color='red')
        plt.plot(d_loss_fake, color='blue')
        plt.title('d_loss')
        fig.add_subplot(1, 3, 2)
        plt.plot(d_acc_real, color='red')
        plt.plot(d_acc_fake, color='blue')
        plt.title('d_acc')
        fig.add_subplot(1, 3, 3)
        plt.plot(g_loss)
        plt.title('g_loss')
        plt.show()

if __name__ == "__main__":
    gan = GAN()
    gan.train(epochs=30000, batch_size=256,save_images_interval=5, save_model_interval=10, interval_history=5)




