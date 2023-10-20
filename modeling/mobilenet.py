import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import cv2
import pathlib
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

dataset_url = r"C:\Users\acorn\Downloads\3part_0616"
data_dir = pathlib.Path(dataset_url)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# 매개변수 정의
batch_size = 16
img_height = 224
img_width = 224

# 데이터셋 분할
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# 훈련 데이터 시각화
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 그레이 스케일 및 에지 감지 함수
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # 그레이스케일 이미지를 RGB로 변환
    edges = cv2.Canny(gray.astype(np.uint8), 80, 150)  # 데이터 타입 변환
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # 엣지 이미지를 RGB로 변환
    return edges.astype(np.float32)

# 데이터 증강 생성기
data_augmentation = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 증강된 이미지 시각화
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation.flow(images, shuffle=False).next()
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].astype("uint8"))
        plt.axis("off")
plt.show()

# 데이터 표준화
normalization_layer = layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# EarlyStopping 생성
es = EarlyStopping()
folder_directory = r'C:\Users\acorn\Desktop\project\\'
checkPoint_path = folder_directory+"mobilenetV3_gray_edge_{epoch}.h5"
chk = ModelCheckpoint(filepath=checkPoint_path, monitor='val_acc', mode='max', verbose=0, save_best_only=True)

# 모델 생성

num_classes = len(class_names)

model = tf.keras.applications.MobileNetV3Small(
    input_shape=(img_height, img_width, 3),
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    include_preprocessing=True,
)

print(model.summary())

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs = 100

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,)
model.save('mobilenetV3_4part_gray_edge_0620.h5')

print("Training Accuracy:", history.history['accuracy'])
print("Validation Accuracy:", history.history['val_accuracy'])
print("Training Loss:", history.history['loss'])
print("Validation Loss:", history.history['val_loss'])

# 모델 시각화

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()