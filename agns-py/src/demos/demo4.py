from tensorflow.keras.models import load_model

# subject n° 142 physical dodging attack -vs- OF 143
if __name__ == '__main__':
    ep = 1
    lr = 5e-5
    weight_decay = 1e-5

    # load model and attacker images
    model = load_model('../../saved-models/OF143.h5')
    target = 142

    # execute attack and shows results
