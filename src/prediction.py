
import torch
import prediction_engine
import siamese_net
import numpy as np

NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
INPUT_SIZE = 2500
OUTPUT_SIZE = 12
CHANNELS=50
KERNEL_SIZES=[50, 10]
STRIDES=[2, 2]
PADDINGS=[1, 1]
HIDDEN_SIZES=[256]

# Setup directories
#test_dir = "datasets/contest_TEST.h5"
test_labels_dir = "datasets/test_labels.csv"
model_dir = 'models/final_model2_dashing_dream_256b_50ep.pth'
test_dir = "datasets/contest_TRAIN.h5"


# Setup target device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create DataLoaders with help from data_setup.py
test_dataloader, y_test = prediction_engine.create_dataloaders(
    test_dir=test_dir,
    test_labels_dir=test_labels_dir,
    batch_size=BATCH_SIZE,
    device = device,
    pred_test = False # USE WITH CAUTION, turn to 'False' if you want to get embeddings of the training data
)


saved_state_dict = torch.load(model_dir, map_location=torch.device('cpu'))

# Create a new instance of your model
model = siamese_net.SiameseNetwork(
    input_size=INPUT_SIZE, 
    output_size=OUTPUT_SIZE, 
    channels=CHANNELS, 
    kernel_sizes=KERNEL_SIZES, 
    strides=STRIDES, 
    paddings=PADDINGS, 
    hidden_sizes=HIDDEN_SIZES
).to(device)
# Load the saved state into the new model instance
model.load_state_dict(saved_state_dict)

#todo save this to a file
prediction_X_test = prediction_engine.predict_test(
                    model=model, 
                    dataloader=test_dataloader,
                    device=device,
                    test_dir=test_dir, 
                    test_labels_dir=test_labels_dir,
                    batch_size=BATCH_SIZE,
                    y_test=y_test
                    )


np.save('datasets/prediction_X_train_dashing_dream.npy', prediction_X_test)        
np.save('datasets/y_train.npy', y_test)          

#https://colab.research.google.com/drive/15D5vAYkhbAs5-txhYTCb_Fp2jiCnHXVN#scrollTo=82F_qINOBbkL
