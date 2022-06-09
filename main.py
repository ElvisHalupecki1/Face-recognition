import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
#
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)
#
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     # Promjena veličine framea na 250x250 kako bi se uslikalo samo lice
#     frame = frame[10:10 + 250, 180:180 + 250, :]
#
#     # Prikupljanje anchor fotografija
#     if cv2.waitKey(1) & 0XFF == ord('a'):
#         # Stvaranje jedinstvenog imena slike
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # Spremanje slike u mapu
#         cv2.imwrite(imgname, frame)
#
#     # Prikupljanje pozitivnih fotografija
#     if cv2.waitKey(1) & 0XFF == ord('p'):
#         # Stvaranje jedinstvenog imena slike
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # Spremanje slike u mapu
#         cv2.imwrite(imgname, frame)
#
#     # Prikaz slike
#     cv2.imshow('Fotografija', frame)
#
#     # Izlazak iz petlje
#     if cv2.waitKey(1) & 0XFF == ord('q'):
#         break
#
# # Gašenje kamere
# cap.release()
# # Zatvaranje svih novootvorenih prozora
# cv2.destroyAllWindows()

#Dohvacanje 300 fotografija iz svake mape
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)

#Oznacavanje pozitivnih fotografija s 1, i negativnih s 0
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

def preprocess(file_path):
    # Čitanje slike iz putanje
    byte_img = tf.io.read_file(file_path)
    # Učitavanje slike
    img = tf.io.decode_jpeg(byte_img)

    # Smanjenje slike na 100x100 piksela
    img = tf.image.resize(img, (100, 100))
    # Smanjenje slike da bude izmedu 0 i 1
    img = img / 255.0


    return img

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img),label)


data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

# samples = data.as_numpy_iterator()
# samp = samples.next()
#
# img = plt.imshow(samp[0])
# plt.show()
# img2 = plt.imshow(samp[1])
# print(samp[2])
# plt.show()

#Training
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


# Testing
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

test_input, test_val, y_true = test_data.as_numpy_iterator().next()

img3 = cv2.subtract(test_input[0],test_val[0])

plt.subplot(1, 4, 1)
plt.imshow(test_input[0])

plt.subplot(1, 4, 2)
plt.imshow(test_val[0])

plt.subplot(1, 4, 3)
plt.imshow(test_input[0][2])

plt.subplot(1, 4, 4)
plt.imshow(test_val[0][2])

plt.show()


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')

embedding = make_embedding()

#embedding.summary()


# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

#siamese_network.summary()

def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()
siamese_model.summary()


binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001



@tf.function
def train_step(batch):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss

from tensorflow.python.keras.metrics import Precision, Recall


def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Creating a metric object
        r = Recall()
        p = Precision()

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx + 1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())


EPOCHS = 50

train(train_data, EPOCHS)
#
#Get a batch of test data

#
#
# Set plot size
#plt.figure(figsize=(10,8))

# i = 0
# for x in range(16):
#     # Set first subplot
#     plt.subplot(1, 4, 1)
#     plt.imshow(test_input[x][0])
#
#     plt.subplot(1, 4, 2)
#     plt.imshow(test_input[x])
#
#     # Set second subplot
#     plt.subplot(1, 4, 3)
#     plt.imshow(test_val[x][0])
#
#     plt.subplot(1, 4, 4)
#     plt.imshow(test_val[x])
#
#
#     print(y_true[x])
#     # Renders cleanly
#     plt.show()


# siamese_model.save('siamesemodelv2.h5')
# siamese_model.compile()
# # Reload model
# siamese_model2 = tf.keras.models.load_model('siamesemodelv2.h5',custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy},compile=True)
#
# #siamese_model2 = tf.saved_model.load('siamesemodelv2.h5')
# # Make predictions with reloaded model
#
# test_input, test_val, y_true = test_data.as_numpy_iterator().next()
#
# y_hat = siamese_model2.predict([test_input, test_val])
# # Post processing the results
# res = []
# print(y_hat)
# for prediction in y_hat:
#     if prediction > 0.5:
#         res.append(1)
#     else:
#         res.append(0)
#
#
# #print(res[0])
# print(res)
# #print(y_true)
#
#
# from tensorflow.python.keras.metrics import Precision, Recall
# r = Recall()
# p = Precision()
#
# for test_input, test_val, y_true in test_data.as_numpy_iterator():
#     yhat = siamese_model2.predict([test_input, test_val])
#     r.update_state(y_true, yhat)
#     p.update_state(y_true,yhat)
#
# print("Recall ", r.result().numpy())
# print("Precision ", p.result().numpy())
#
# siamese_model2.predict([test_input, test_val])
#
# siamese_model2.summary()


os.listdir(os.path.join('application_data', 'verification_images'))
os.path.join('application_data', 'input_image', 'input_image.jpg')

for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    print(validation_img)


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[10:10 + 250, 180:180 + 250, :]

    cv2.imshow('Verification', frame)

    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder
        #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #         h, s, v = cv2.split(hsv)

        #         lim = 255 - 10
        #         v[v > lim] = 255
        #         v[v <= lim] -= 10

        #         final_hsv = cv2.merge((h, s, v))
        #         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(siamese_model, 0.5, 0.5)
        print(verified)
        print(results)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()