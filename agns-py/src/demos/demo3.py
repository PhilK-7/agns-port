from tensorflow.keras.models import load_model

# subject n° 6 physical impersonation attack -vs- VGG 10
if __name__ == '__main__':
    ep = 1
    lr = 5e-5
    weight_decay = 1e-5

    # load model and attacker images
    model = load_model('../../saved-models/VGG10.h5')
    target = 6

    # execute attack and shows results
