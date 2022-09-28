from tensorflow.keras.models import load_model
import numpy as np

NUM_PRIOR = 100
NUM_PRO = 8
# read posterior response from DSI
term = np.load('d_posterior.npz')
posterior_d = term['arr_0']
print('posterior_d: ', posterior_d.shape)

# Rearrange as: P1：WOPR、WWPR； P2：WOPR、WWPR； ...
posterior = []
for i in range(len(posterior_d[0])):
    temp = []
    for j in range(len(posterior_d)):
        temp.append(posterior_d[j][i])
    posterior.append(temp)
posterior = np.array(posterior)
posterior = np.reshape(posterior, (NUM_PRIOR, 50*NUM_PRO))
print('posterior:', posterior.shape)

for i in range(5):
    # load model
    file = 'model_0'+str((i+1))+'0000.h5'
    model = load_model(file)
    # save as compressed numpy array
    perm = model.predict(posterior)
    perm_gan = perm.reshape(NUM_PRIOR, 3600)
    filename = 'perm_gan_'+ str(i+1) +'00.npz'
    np.savez_compressed(filename, perm_gan)
    print('Saved dataset: ', filename)
