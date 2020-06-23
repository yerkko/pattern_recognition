import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

BASE_FOLDER = 'FaceMask166'

def load_image(image_name, gray=False, crop=False, base_folder=BASE_FOLDER):
    """Carga una imagen con open-cv."""
    if not image_name.endswith('.jpg'):
        image_name = f'{image_name}.jpg'
    
    image_path = os.path.join(base_folder, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise Exception(f'Cant load image: {image_name}')
    
    # open-cv loads as BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if crop:
        image = image[:128, :]

    return image


def load_dataset(n_images, gray=True, crop=True):
    """Carga un dataset de las primeras N imagenes.
    
    Returns:
        dataset lista de: [image_cropped, image_name, i_image (label), j (image_number)]
    """
    dataset = []
    image_base_name = 'F'
    for i_image in range(1, n_images+1):
        base_name = f'FM{i_image:06d}'

        for j in range(1, 6+1):
            # Load image
            image_name = f'{base_name}_{j:02d}'
            image = load_image(image_name, gray=gray, crop=crop)
            
            dataset.append((image, image_name, i_image, j))
            
    return dataset

def create_df(dataset, merge=None):
    """Crea un dataframe a partir de un dataset.
    
    Args:
        dataset -- formato que retorna `load_dataset()`
    """
    data = [sample[2:] for sample in dataset]
    image_names = [sample[1] for sample in dataset]
    df = pd.DataFrame(data, index=image_names, columns=['label', 'image_number'])
    
    if merge is not None:
        df = df.merge(merge, right_index=True, left_index=True)

    return df


def calculate_feature_df(dataset, feature_fn, feature_name, **kwargs):
    """Calcula un feature con `feature_fn()` para todas las imagenes de un dataset, retorna un DF.
    
    Args:
        dataset -- formato que retorna `load_dataset()`
        feature_fn -- funcion de firma `features = feature_fn(image, **kwargs)`
        feature_name -- string para usar como nombre base del feature
        **kwargs -- entregados a `feature_fn()`
    """
    print(f'Calculating {feature_name}')
    image_names = []
    features = []
    
    # Cargar imagenes y calcular vectores de features
    for sample in dataset:
        image = sample[0]
        image_names.append(sample[1])

        image_features = feature_fn(image, **kwargs)
        features.append(image_features)
    
    features = np.array(features)
    n_images, n_feats = features.shape

    # Nombres de los features, e.g. 'hog_0', 'hog_1', etc
    columns = [f'{feature_name}_{idx}' for idx in range(n_feats)]
    
    # Crear DataFrame, el index son nombres de las imagenes
    feature_df = pd.DataFrame(features, index=image_names, columns=columns)
    
    # Retornar DataFrame con features calculados
    return feature_df


def plot_cm(cm, title='', labels=None, colorbar=False):
    """Plotear una matriz de confusion.
    
    Args:
        cm -- matriz de n_classes x n_classes
        title -- string para el titulo
        labels -- lista de nombres de n_classes
        colorbar -- si incluir la barra de color o no
    """
    n_labels = len(cm)
    ticks = np.arange(n_labels)
    
    if labels is None:
        labels = ticks

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    if colorbar:
        plt.colorbar()

    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.title(title)
    plt.xlabel('Predicción')
    plt.ylabel('Real')

    thresh = cm.max() / 2
    for row in range(n_labels):
        for col in range(n_labels):
            value = cm[row, col]
            color = 'white' if value > thresh else 'black'

            value_str = f'{int(value):d}'

            plt.text(col, row, value_str, ha='center', va='center', color=color)
    plt.show()


def normalize_df(df, keep_nonorm=True, ignore=['label', 'image_number']):
    """Normaliza los datos de un DataFrame.
    
    Args:
        df -- DataFrame con shape (n_imagenes, n_features)
        keep_nonorm -- Si es True, devuelve el dataframe incluyendo las columnas
            normalizadas y no normalizadas
        ignore -- set de columnas que NO son features
    Returns:
        DataFrame con columnas de features normalizadas
    """
    # Filter to normalize only with train data
    df_train = df.loc[df['image_number'] < 4]

    # Use only features
    feature_columns = [col for col in df.columns if col not in ignore and '_nonorm' not in col]

    # Fit normalization with training
    normalization = MinMaxScaler(feature_range=(0, 1))
    normalization.fit(df_train[feature_columns])

    # Normalize whole dataset
    data_norm = normalization.transform(df[feature_columns])
    df_norm = pd.DataFrame(data_norm, index=df.index, columns=feature_columns)


    # Keep both columns (normalized and not normalized)
    if keep_nonorm:
        # Rename no-normalized to '<col>_nonorm'
        rename = {col: f'{col}_nonorm' for col in feature_columns}
        df_nonorm = df.rename(columns=rename, inplace=False)

        # Merge norm and nonorm
        return df_nonorm.merge(df_norm, how='inner', right_index=True, left_index=True)

    # Add the ignored columns from the original dataset
    return df_norm.merge(df[ignore], how='inner', right_index=True, left_index=True)
