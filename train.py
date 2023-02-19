import torchaudio
import torch
from torch import nn
from torch.utils.data import DataLoader
from preprocessing import UrbanSoundDataset
from build_model import Conv


BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "model.pth"


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predicitons = model(inputs)
        loss = loss_fn(predicitons, targets)

        # backprop loss, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}\n")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-------------------------")
    print("Finished training")

    



if __name__ ==  "__main__":

    d_path = "data/UrbanSound8K"
    ANNOTATIONS_FILE = f"{d_path}/metadata/UrbanSound8K.csv"
    AUDIO_DIR = f"{d_path}/audio"
    SR = 22050
    NUM_SAMPLES = 22050
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=1024,
        hop_length=512, 
        n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, 
        AUDIO_DIR, 
        transformation=mel_spectogram, 
        target_sample_rate=SR, 
        num_samples = NUM_SAMPLES,
        device=device
    )
    
    # create dataloader
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    cnn = Conv().to(device)
    
    # loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train the model
    train(cnn, train_data_loader, loss_fn, optimizer, device, epochs=EPOCHS)


    # STORE
    torch.save(cnn.state_dict(), MODEL_PATH)

    print(f"model saved at {MODEL_PATH}")

    

