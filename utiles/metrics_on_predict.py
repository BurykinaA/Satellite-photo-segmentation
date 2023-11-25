from PIL import Image
import numpy as np
from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score, precision_score, recall_score
from segmentation_models_pytorch.utils.metrics import IoU, Fscore

# Загрузите изображения
image1 = Image.open("/home/jupyter/datasphere/project/answers/new.png")
image2 = Image.open("/home/jupyter/datasphere/project/data/mask/train_mask_020.png")

# Преобразуйте изображения в массивы numpy
image1_np = np.array(image1)
image2_np = np.array(image2)

# unique_values_image1 = np.unique(image1_np)
# unique_values_image2 = np.unique(image2_np)

# print(f'Уникальные значения в первом массиве: {unique_values_image1}')
# print(f'Уникальные значения во втором массиве: {unique_values_image2}')

# Предположим, что изображения являются метками классов, преобразуйте их в одномерные массивы
image1_np = image1_np.flatten()
image2_np = image2_np.flatten()

# Вычислите F1 macro и micro score
f1_macro = f1_score(image1_np, image2_np, average="macro")
f1_micro = f1_score(image1_np, image2_np, average="micro")
f1 = f1_score(image1_np, image2_np)

print(f"F1 Score: {f1}")
print(f"F1 Macro Score: {f1_macro}")
print(f"F1 Micro Score: {f1_micro}")


# Вычислите основные метрики бинарной классификации
accuracy = accuracy_score(image1_np, image2_np)
precision = precision_score(image1_np, image2_np)
recall = recall_score(image1_np, image2_np)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Вычислите IoU и Dice score
iou = IoU()(torch.tensor(image1_np), torch.tensor(image2_np))
dice = Fscore()(torch.tensor(image1_np), torch.tensor(image2_np))

print(f"IoU: {iou}")
print(f"Dice: {dice}")
