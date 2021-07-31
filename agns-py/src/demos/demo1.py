from tensorflow.keras.models import load_model

# subject n° 19 digital dodging attack -vs- VGG143
if __name__ == '__main__':
    ep = 1
    lr = 5e-5
    weight_decay = 1e-5

    # TODO load model and attacker images
    model = load_model('../../saved-models/VGG143.h5')
    target = 19

    # TODO execute attack and show results
