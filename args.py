IMAGE_SIZE = 224  # Image size (224x224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]  # Std of ImageNet dataset (used for normalization)
BATCH_SIZE = 8
LEARNING_RATE = 0.001
LEARNING_RATE_SCHEDULE_FACTOR = 0.1  # Parameter used for reducing learning rate
LEARNING_RATE_SCHEDULE_PATIENCE = 5  # Parameter used for reducing learning rate
MAX_EPOCHS = 100

TRAIN_PATH = './train.csv'
VAL_PATH = './valid.csv'

LABELS_TO_PREDICT = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion']
LABELS_ENCODING = ['LabelOne', 'LabelZero', 'LabelMulti']

NUM_CLASSES = 4

