import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Input, Flatten, Conv2D, Conv2DTranspose
from keras.optimizers import Adam

# MNIST veri kümesini yükleme
(X_train, _), (_, _) = mnist.load_data()

# Veri önişleme
X_train = X_train.astype('float32') / 255
X_train = np.expand_dims(X_train, axis=-1)  # Resimlerin kanal boyutunu ekleyin

# GAN için gürültü boyutu
noise_dim = 100

# Oluşturucu ağı
def build_generator():
    model = Sequential()
    model.add(Dense(128 * 7 * 7, input_dim=noise_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same', activation='sigmoid'))
    noise = Input(shape=(noise_dim,))
    img = model(noise)
    return Model(noise, img)

# Ayırt edici ağı
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    img = Input(shape=(28, 28, 1))
    validity = model(img)
    return Model(img, validity)

# Derleme ve eğitim
optimizer = Adam(0.0002, 0.5)

# Derleme
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

generator = build_generator()
z = Input(shape=(noise_dim,))
img = generator(z)

# Sadece oluşturucuyu eğitmek için ayarla
discriminator.trainable = False

# Sahte görüntülerin oluşturulması
valid = discriminator(img)

# Birleştirilmiş model (Güncelleme sırasında yalnızca oluşturucuyu eğitiyoruz)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# GAN modelinin eğitimi
def train(epochs, batch_size, sample_interval):
    # Etiketler
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Rastgele bir örnek alın ve oluşturucu tarafından bir görüntü oluşturun
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))

        # Görüntü üretme
        gen_imgs = generator.predict(noise)

        # Eğitim
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Oluşturucuyu eğitin
        g_loss = combined.train_on_batch(noise, valid)

        # İlerleme gösterme
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
            # Üretilen resimleri kaydetme
            sample_images(epoch)

# Üretilen örnek resimleri kaydetme
def sample_images(epoch):
    os.makedirs("images", exist_ok=True)  # Dizin oluşturuldu
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, noise_dim))
    gen_imgs = generator.predict(noise)

    # Resimleri düzenleme
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"images/mnist_{epoch}.png")
    plt.close(fig)


# Eğitim
epochs = 20000
batch_size = 128
sample_interval = 100

train(epochs, batch_size, sample_interval)
