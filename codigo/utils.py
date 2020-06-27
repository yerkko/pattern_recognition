import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

from pybalu.feature_transformation import pca
from pybalu.feature_selection import sfs
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


def calculate_features(dataset, feature_fn, max_images=None, show=False, **kwargs):
    """Calcula features para imagenes de un dataset.
    
    Nota: esta funcion podria ser reutilizada en calculate_features_df()
    """
    print(f'Calculating features')
    features = []
    
    if max_images is None:
        data = dataset
    else:
        data = dataset[:max_images]

    # Cargar imagenes y calcular vectores de features
    for sample in tqdm(data, disable=not show):
        image = sample[0]

        image_features = feature_fn(image, **kwargs)
        features.append(image_features)
    
    return features


def features_to_df(dataset, features, feature_name):
    """Transforma un array de features a un DataFrame.
    
    Nota: esta funcion podria ser reutilizada en calculate_feature_df()
    
    Args:
        dataset -- formato que retorna `load_dataset()`
        features -- np.array de shape n_images, n_features
        feature_name -- string que se usa como nombre base.
        
    Returns:
        dataframe
    """
    image_names = [sample[1] for sample in dataset]
    
    n_images, n_feats = features.shape

    columns = [f'{feature_name}_{idx}' for idx in range(n_feats)]
    
    feature_df = pd.DataFrame(features, index=image_names, columns=columns)

    return feature_df



def calculate_feature_df(dataset, feature_fn, feature_name, max_images=None, show=False,
                         fix_len_mode='pad',
                         **kwargs):
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
    
    if max_images is None:
        data = dataset
    else:
        data = dataset[:max_images]

    # Cargar imagenes y calcular vectores de features
    for sample in tqdm(data, disable=not show):
        image = sample[0]
        image_names.append(sample[1])

        image_features = feature_fn(image, **kwargs)
        features.append(image_features)
    
    # Pass to numpy
    features = np.array(features)
    
    
    # DEPRECATED: see new functions for SIFT and SURF features    
    #     if len(features.shape) == 1 and len(features[0].shape) == 1:
    #         # HACK: assume that in this case the image_features have different lengths, so pad to the greatest
    #         if fix_len_mode == 'pad':
    #             max_len = max(len(f) for f in features)
    #             features = np.array([
    #                 np.pad(f, pad_width=(0, max_len-len(f)), mode='constant', constant_values=0) for f in features
    #             ])
    #         elif fix_len_mode == 'crop':
    #             min_len = min(len(f) for f in features)
    #             features = np.array([f[:min_len] for f in features])

    
    if len(features.shape) != 2:
        print('Cannot parse to DF')
        # HACK: for cases where image_features have different lengths
        return features

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
        labels = ticks + 1

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


def PCA(x_train, x_test, x_val, n_components):
    """
    Realiza la transformación PCA de los datos a tan solo 'n_components' características.
    n_components:   número de características.
    """
    x_train, _, A, Xm, _ = pca(x_train, n_components=n_components)
    x_test = np.matmul(x_test - Xm, A)
    x_val = np.matmul(x_val - Xm, A)
    return x_train, x_test, x_val


def SFS(x_train, y_train, x_test, x_val, n_features, method="fisher", show=False):
    """
    Realiza la selección de 'n_features' características secuencialmente.
    n_features:     número de características
    method:         método a ocupar en la selección
    show:           mostrar el progreso de la selección.
    """
    s_sfs = sfs(x_train, y_train, n_features=n_features, method="fisher", show=show)
    x_train = x_train[:, s_sfs]
    x_test = x_test[:, s_sfs]
    x_val = x_val[:, s_sfs]
    return x_train, x_test, x_val
