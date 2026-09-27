from scripts.train_model import make_synthetic_data


def test_synthetic_training_data_is_reproducible_and_valid():
    X1, y1 = make_synthetic_data(rows=100, seed=9)
    X2, y2 = make_synthetic_data(rows=100, seed=9)
    assert X1.equals(X2)
    assert (y1 == y2).all()
    assert len(X1) == len(y1) == 100
    assert set(X1.weather.unique()) <= {"clear", "rain", "wind", "heat"}
