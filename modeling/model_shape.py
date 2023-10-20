from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# 모델 로드
model = load_model('resnet50_gray_edge_0619.h5')

# 모델 구조 저장
plot_model(model, to_file='model.png', show_shapes=True)