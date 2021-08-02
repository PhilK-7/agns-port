from tensorflow.keras.models import load_model

# subject n° 1 digital impersonation attack -vs- OpenFace 10
if __name__ == '__main__':
    ep = 1
    lr = 5e-5
    weight_decay = 1e-5

    # load model and attacker images
    model = load_model('../saved-models/of10.h5')
    target = 1

    # execute attack and shows results
