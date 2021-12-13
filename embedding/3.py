import numpy as np

entity_emb = np.load('entity_embedding0.npy')
rel_emb = np.load('relation_embedding0.npy')

np.savetxt('entity_embedding0.txt', entity_emb, fmt='%.15f')
np.savetxt('relation_embedding0.txt', rel_emb, fmt='%.15f')
