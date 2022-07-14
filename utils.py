import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
import args as args
from sklearn.model_selection import train_test_split

def labelone(row):
    labelone = []
    for feature in row[1:5].index.values:
        if row[feature] in [1, -1]:
            labelone.append(1)
        else:
            labelone.append(0)
    return labelone


def labelzero(row):
    labelzero = []
    for feature in row[1:5].index.values:
        if row[feature] in [1]:
            labelzero.append(1)
        else:
            labelzero.append(0)
    return labelzero


def labelmulti(row):
    labelmulti = []
    for feature in row[1:5].index.values:
        if row[feature] in [1, 0, -1]:
            labelmulti.append(int(row[feature]))
        else:
            labelmulti.append(0)
    return labelmulti


def get_labels(df):
    df = df.copy()
    df.loc[:, 'LabelOne'] = df.apply(labelone, axis=1)
    df.loc[:, 'LabelZero'] = df.apply(labelzero, axis=1)
    df.loc[:, 'LabelMulti'] = df.apply(labelmulti, axis=1)
    return df


def select_cols(df, cols):
    df = df[cols]
    return df


def drop_all_zeros(row):
    if row[-1] == [0, 0, 0, 0]:
        return row
    

def select_non_zeros(df):
    rows_to_drop = df.apply(lambda x: drop_all_zeros(x), axis=1)
    df = df.drop(rows_to_drop.dropna().index).reset_index(drop=True)
    return df


def dataloader():
    train_df = pd.read_csv(args.TRAIN_PATH)
    test_df = pd.read_csv(args.VAL_PATH)
    
    train_df = train_df[:80000]

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    #train_df = train_df[:2000]

    train_df = select_non_zeros(train_df)
    val_df = select_non_zeros(val_df)
    test_df = select_non_zeros(test_df)

    cols = ['Path'] + args.LABELS_TO_PREDICT
    train_df = select_cols(train_df, cols)
    val_df = select_cols(val_df, cols)
    test_df = select_cols(test_df, cols)

    train_df = get_labels(train_df)
    val_df = get_labels(val_df)
    test_df = get_labels(test_df)

    cols = ['Path'] + args.LABELS_ENCODING
    train_df = select_cols(train_df, cols)
    val_df = select_cols(val_df, cols)
    test_df = select_cols(test_df, cols)

    return train_df, val_df, test_df


class ChestXrayDataset(Dataset):
    def __init__(self, folder_dir, dataframe, image_size, normalization, mapping='u-ones'):
        """
        Init Dataset

        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe: pandas.DataFrame
            dataframe contains all information of images
        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """
        self.image_paths = []  # List of image paths
        self.image_labels = []  # List of image labels
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.image_size = image_size
        self.normalization = normalization
        self.mapping = mapping

        self.image_transformation = self.transformations()

        self.dataparser()

    def transformations(self):
        image_transformation = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ]

        if self.normalization:
            # Normalization with mean and std from ImageNet
            image_transformation.append(transforms.Normalize(args.IMAGENET_MEAN, args.IMAGENET_STD))

        return transforms.Compose(image_transformation)

    def dataparser(self):
        for index, row in self.dataframe.iterrows():
            image_path = os.path.join(self.folder_dir, row.Path)
            self.image_paths.append(image_path)
            if self.mapping == 'u-ones':
                labels = row.LabelOne
            elif self.mapping == 'u-zero':
                labels = row.LabelZero
            elif self.mapping == 'u-multi':
                labels = row.LabelMulti
            self.image_labels.append(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """

        # Read image
        image_path = self.image_paths[index]
        image_data = Image.open(r'%s' % image_path).convert("RGB")  # Convert image to RGB channels

        # TODO: Image augmentation code would be placed here

        # Resize and convert image to torch tensor
        image_data = self.image_transformation(image_data)

        return image_data, torch.FloatTensor(self.image_labels[index])


def multi_label_auroc(y_gt, y_pred):
    """ Calculate AUROC for each class

    Parameters
    ----------
    y_gt: torch.Tensor
        groundtruth
    y_pred: torch.Tensor
        prediction

    Returns
    -------
    list
        F1 of each class
    """
    auroc = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = y_pred.to("cpu").numpy()
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        auroc.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return auroc


def multi_label_accuracy(y_gt, y_pred):
    """ Calculate AUROC for each class

    Parameters
    ----------
    y_gt: torch.Tensor
        groundtruth
    y_pred: torch.Tensor
        prediction

    Returns
    -------
    list
        F1 of each class
    """
    acc = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = y_pred.to("cpu").numpy()
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        acc.append(accuracy_score(gt_np[:, i], pred_np.round()[:, i]))
    return acc
