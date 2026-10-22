"""抽特征按同一帧调用检测和手部。"""
import numpy as np

from lstm_workbench.extract import rows_from_frames
from lstm_workbench.features import feature_size


def test_same_frame_detection_and_hands_become_one_row():
    frame = np.zeros((20, 40, 3), dtype=np.uint8)

    def detect(_frame):
        return [{"label": "part", "conf": 0.95, "box": (0, 0, 10, 10)}]

    def hands(_frame):
        return [[(4, 2, 0.1)] + [(0, 0, 0)] * 20]

    array = rows_from_frames([frame, frame], detect, hands, ["part"], 0.5)
    assert array.shape == (2, feature_size(1))
    assert array[0, 0] == 0.95
    assert array[0, 5] == 4 / 40
