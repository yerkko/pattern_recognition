"""Funciones de utilidad para correr experimentos."""

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import RFECV, SelectKBest, chi2
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from utils import plot_cm


COMMON_STRATEGIES = [
    { 'method': 'SVM', 'kernel': 'rbf' },
    { 'method': 'SVM', 'kernel': 'linear' },
    { 'method': 'KNN', 'n_neighbors': 5 },
    { 'method': 'KNN', 'n_neighbors': 5, 'weights': 'distance' },
    { 'method': 'KNN', 'n_neighbors': 3 },
    { 'method': 'KNN', 'n_neighbors': 3, 'weights': 'distance' },
    { 'method': 'KNN', 'n_neighbors': 1 },
    { 'method': 'MLP' },
    { 'method': 'RF' },
    { 'method': 'LDA' },
]


def split_train_val_test(df):
    """Se hace split de un dataframe en train, val, test.
    
    Args:
        df -- formato retornado por `create_df()`
    """
    train = df.loc[df['image_number'] < 4]
    val = df.loc[df['image_number'] == 4]
    test = df.loc[df['image_number'] > 4]
    return train, val, test


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, cm


def run_experiment(dataset_df, train_cols,
                   method='KNN', n_neighbors=3, hidden_layer_sizes=(100,),
                   option='validation', n_trees = 100, ensemble = False, show_cm=True, **kwargs):
    if len(train_cols) == 0:
        raise Exception('No columns provided to run_experiment()')
    
    # Split train, val, test
    train, val, test = split_train_val_test(dataset_df)
    
    # Seleccionar columnas para entrenar
    x_train = train[train_cols]
    y_train = train['label']

    if option == 'validation' or option == 'val':
        x_val = val[train_cols]
        y_val = val['label']
    elif option == 'test':
        x_val = test[train_cols]
        y_val = test['label']
    else:
        raise Exception(f'Option not recognized: {option}')

    
    # Elegir clasificador
    if method == 'KNN':
        model = KNN(n_neighbors=n_neighbors, **kwargs)
    elif method == 'SVM':
        model = SVC(**kwargs)
    elif method == 'MLP':
        model = MLP(hidden_layer_sizes=hidden_layer_sizes, **kwargs)
    elif method == 'RF':
        model = RandomForestClassifier(n_estimators = n_trees, **kwargs)
    elif method == 'LDA':
        model = LDA(**kwargs)
    else:
        raise Exception(f'Unkwown model: {method}')
        
    if ensemble:
        model = BaggingClassifier(model, max_samples = 0.5, max_features = 0.5)
        
    # Entrenar clasificador
    print('Training...')
    model.fit(x_train, y_train)
    
    # Evaluar modelo
    train_accuracy, train_cm = evaluate(model, x_train, y_train)
    val_accuracy, val_cm = evaluate(model, x_val, y_val)
    
    print(f'Accuracy: train: {train_accuracy}, {option}: {val_accuracy}')
    if show_cm:
        plot_cm(val_cm, title=f'{option} CM')

    return val_accuracy


def find_best_strategy(df, cols, strategies, option='val'):
    """Corre experimentos con distintas estrategias, y retorna la mejor estrategia.
    
    Args:
        df -- dataframe de un dataset
        cols -- set de columnas a usar para el experimento
        strategies -- lista de estrategias: diccionario con argumentos a pasar a `run_experiment()`
            (ver ejemplos mas abajo)
        option -- con que set testear (val o test)
    """
    results = []
    
    for strategy in strategies:
        print(strategy, end='\t', flush=True)
        acc = run_experiment(df, cols, option=option, show_cm=False, **strategy)

        results.append((acc, strategy))

    results = sorted(results, key=lambda x:x[0], reverse=True)
    best_acc, best_strategy = results[0]
    # print('Best result: ', best_acc, best_strategy)

    return best_acc, best_strategy
