import pandas as pd
from eda import get_size, extract_features

def test_get_size():
    """
    Tests whether get_size returns correct dimensions.
    """
    data = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    assert get_size(data) == (3, 2)
def test_extract_features():
    """
    Tests feature extraction using known measurements.
    """
    data = pd.DataFrame({
        0: [2, 10],
        1: [4, 10],
        2: [6, 10]
    })

    result = extract_features(data, "PS1")

    assert result.loc[0, "PS1_mean"] == 4
    assert result.loc[0, "PS1_std"] == 2
    assert result.loc[0, "PS1_max"] == 6

    assert result.loc[1, "PS1_mean"] == 10
    assert result.loc[1, "PS1_std"] == 0
    assert result.loc[1, "PS1_max"] == 10

def main():
    test_get_size()
    test_extract_features()

    print("All tests passed!")


if __name__ == "__main__":
    main()