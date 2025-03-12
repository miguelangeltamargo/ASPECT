import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from collections import Counter
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay


def count_plot(x, title, dir):
    """
    Plot the class distribution of the data set set

    Parameters
    ----------
    Y : numpy array
        The class labels of the data set
    label : str
        The name of the data

    Returns
    -------
    None
    """

    label_counts = Counter(x)
    label_counts_df = pd.DataFrame.from_dict(label_counts, orient='index').reset_index()
    label_counts_df.columns = ['Label', 'Count']
    sns.barplot(x='Label', y='Count', hue='Label', data=label_counts_df, palette='Set1', legend=False)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(f'{title}')
    pathf = dir / 'graphs'
    os.makedirs(pathf, exist_ok=True)
    filename = f'{title}.png'
    plt.savefig(os.path.join(pathf, filename) )
    plt.show()

def plot_tsne(X, y, title):
    """
    Plot the TSNE transform of X colored by y

    Parameters
    ----------
    X : numpy array
        The data to be transformed. It can be extracted features by a model e.g. DNA-BERT
    y : numpy array
        The class labels of the data

        
    Returns
    -------
    None
    """

    X_tsne = TSNE(n_components=2, n_iter = 2000 , init='random').fit_transform(X)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.title(f'TSNE - {title} data 2D')
    plt.legend(title)
    plt.savefig(f'{title}.png')
    plt.show()    

def plot_pca(X, y, title):
    """
    Plot the PCA transform of X colored by y
    
    Parameters
    ----------
    X : numpy array
        The data to be transformed. It can be extracted features by a model e.g. DNA-BERT
    y : numpy array
        The class labels of the data
    
    Returns
    -------
    None
    """
    
    # create a PCA object with 2 components
    pca = PCA(n_components=2)

    # fit the PCA object to X and transform X
    X_pca = pca.fit_transform(X)

    # plot the PCA transform of X colored by y
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f'{title}.png')
    plt.show()
    

# Define function to plot confusion matrix
def plot_confusion_matrix(trainer, eval_dataset, results_dir, dataset):
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Constitutive", "Cassette"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix {dataset}")
    plt.savefig(results_dir / f"confusion_matrix_{dataset}.png")
    plt.close()
    
    # Define function to plot confusion matrix FOR NO FOLDS ONLY TRIALS
def plot_trial_confusion_matrix(trainer, eval_dataset, trial, results_dir):
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Constitutive", "Cassette"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix Trial {trial}")
    plt.savefig(results_dir / f"confusion_matrix_Trial_{trial}.png")
    plt.close()