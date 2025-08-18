import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tiger.data.sentence_embedding import SentenceEmbeddingsDataset
from tiger.models.rqvae import RQVAE

BATCH_SIZE = 1024
LR = 0.4
NUM_EPOCHS = 20_000
BETA = 0.24
CODEBOOK_SIZE = 256
LATENT_DIM = 32

embeddings = np.load("./data/processed/2014/Beauty_sentence_embeddings.npy")
dataset = SentenceEmbeddingsDataset(torch.from_numpy(embeddings))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = RQVAE()
optimiser = optim.Adagrad(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    if epoch == 0:
        init_batch = next(iter(dataloader))
        model.initialize_codebooks(init_batch)

    for batch in dataloader:
        optimiser.zero_grad()
        x_hat, loss = model(batch)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(dataloader)}")
