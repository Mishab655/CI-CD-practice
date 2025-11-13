import joblib
from train import train

def test_training():
    train()
    model = joblib.load('model.pkl')
    assert model is not None
