"""整条视频审核完成后，空白区间才允许作为空闲训练。"""
from copy import deepcopy

import pytest

from lstm_workbench import service
from lstm_workbench.project_store import ProjectStore


def payload(complete=True):
    return {
        "classes": [{"id": "pack", "name": "包装"}],
        "clips": [{
            "id": name, "file": f"{name}.mp4", "frame_count": 80,
            "fps": 10, "width": 16, "height": 8,
            "annotation_complete": complete,
            "segments": [{"id": name + "-s", "class_id": "pack", "start": 0, "end": 40}],
            "excluded": [],
        } for name in ("a", "b")],
    }


SETTINGS = {"split_mode": "clip", "val_clip_ids": ["b"], "window_seconds": 1.92}


def test_legacy_and_incomplete_participating_clips_block_training():
    raw = payload(False)
    del raw["clips"][0]["annotation_complete"]
    reasons = service.annotation_blockers(raw, SETTINGS)
    assert len([r for r in reasons if "标注完成" in r]) == 2
    assert service.annotation_blockers(payload(), SETTINGS) == []


def test_unchecked_clip_does_not_add_completion_blocker():
    raw = payload()
    raw["clips"].append({**raw["clips"][0], "id": "unchecked", "annotation_complete": False})
    assert service.annotation_blockers(raw, {**SETTINGS, "train_excluded": ["unchecked"]}) == []


@pytest.mark.parametrize("value", ["true", 1, None, [], {}])
def test_completion_value_requires_a_real_boolean(value):
    raw = payload()
    raw["clips"][0]["annotation_complete"] = value
    assert "布尔" in service.validate_annotations(raw)


def test_content_change_revokes_confirmation_and_preserves_other_clips():
    before = payload()
    draft = deepcopy(before)
    draft["clips"][0]["segments"][0]["end"] = 41
    saved = service.apply_annotation_completion(before, draft)
    assert saved["clips"][0]["annotation_complete"] is False
    assert saved["clips"][1]["annotation_complete"] is True
    assert before["clips"][0]["segments"][0]["end"] == 40


@pytest.mark.parametrize("field,value", [
    ("excluded", [{"id": "cut", "start": 50, "end": 60}]),
    ("frame_count", 81), ("file", "replaced.mp4"), ("fps", 11),
])
def test_media_or_exclusion_change_revokes_confirmation(field, value):
    before = payload()
    draft = deepcopy(before)
    draft["clips"][0][field] = value
    assert service.apply_annotation_completion(before, draft)["clips"][0]["annotation_complete"] is False


def test_explicit_confirmation_can_confirm_edited_content_in_one_save():
    before = payload()
    draft = deepcopy(before)
    draft["clips"][0]["segments"][0]["end"] = 41
    draft["completion_confirmations"] = ["a"]
    saved = service.apply_annotation_completion(before, draft)
    assert saved["clips"][0]["annotation_complete"] is True
    assert "completion_confirmations" not in saved


def test_renaming_classes_preserves_completion_but_removed_actions_revoke_it():
    before = payload()
    draft = deepcopy(before)
    draft["classes"][0]["name"] = "包装步骤"
    assert all(c["annotation_complete"] for c in service.apply_annotation_completion(before, draft)["clips"])
    draft["clips"][0]["segments"] = []
    assert service.apply_annotation_completion(before, draft)["clips"][0]["annotation_complete"] is False


def test_new_clip_is_never_confirmed_without_explicit_confirmation():
    saved = service.apply_annotation_completion({"classes": [], "clips": []}, payload())
    assert all(not c["annotation_complete"] for c in saved["clips"])


def test_unchanged_clip_can_be_confirmed_and_explicit_false_is_respected():
    before = payload(False)
    draft = payload()
    assert all(c["annotation_complete"] for c in service.apply_annotation_completion(before, draft)["clips"])
    draft["clips"][0]["annotation_complete"] = False
    draft["completion_confirmations"] = ["a"]
    assert not service.apply_annotation_completion(before, draft)["clips"][0]["annotation_complete"]


def test_missing_field_preserves_an_existing_confirmation_and_null_exclusions_are_safe():
    before = payload()
    before["clips"][0]["excluded"] = None
    draft = deepcopy(before)
    del draft["clips"][0]["annotation_complete"]
    assert service.apply_annotation_completion(before, draft)["clips"][0]["annotation_complete"] is True


def test_bad_confirmation_list_is_rejected():
    for bad in ("a", ["missing"], [1], ["a", "a"]):
        raw = payload()
        raw["completion_confirmations"] = bad
        assert service.validate_annotations(raw)


def test_adopted_clip_starts_unconfirmed(tmp_path):
    store = ProjectStore(tmp_path)
    store.create("demo", {})
    service.adopt_clip(store, "demo", "a.mp4", {"frame_count": 10, "fps": 10, "width": 16, "height": 8}, replace=False)
    assert store.load_annotations("demo")["clips"][0]["annotation_complete"] is False


def test_http_save_confirms_content_and_revokes_stale_confirmation(tmp_path):
    from lstm_workbench.tests.test_app import running

    store = ProjectStore(tmp_path)
    store.create("demo", {})
    store.save_annotations("demo", payload(False))
    with running(tmp_path) as call:
        draft = payload()
        assert call("POST", "/api/projects/demo/annotations", draft)[0] == 200
        assert store.load_annotations("demo")["clips"][0]["annotation_complete"] is True
        draft["clips"][0]["segments"][0]["end"] = 41
        assert call("POST", "/api/projects/demo/annotations", draft)[0] == 200
        saved = store.load_annotations("demo")
        assert saved["clips"][0]["annotation_complete"] is False
        assert saved["clips"][1]["annotation_complete"] is True
        draft["completion_confirmations"] = ["a"]
        assert call("POST", "/api/projects/demo/annotations", draft)[0] == 200
        saved = store.load_annotations("demo")
        assert saved["clips"][0]["annotation_complete"] is True
        assert "completion_confirmations" not in saved


def test_http_training_refuses_incomplete_annotations_before_extracting(tmp_path, monkeypatch):
    from lstm_workbench import app
    from lstm_workbench.tests.test_app import running

    store = ProjectStore(tmp_path)
    store.create("demo", SETTINGS)
    store.save_annotations("demo", payload(False))
    def unexpected(*args, **kwargs):
        raise AssertionError("不应抽取特征")
    monkeypatch.setattr(app, "_extract_tracked", unexpected)
    with running(tmp_path) as call:
        code, response = call("POST", "/api/projects/demo/train", {})
        assert code == 400 and "标注完成" in response["error"]
        assert call("POST", "/api/projects/demo/train", {})[0] == 400, "失败后任务锁必须释放"


def test_background_preparation_still_works_without_completion(tmp_path, monkeypatch):
    from lstm_workbench.tests.test_preparation import setup_project, wait_for

    wb, calls = setup_project(tmp_path, monkeypatch)
    raw = wb.store.load_annotations("demo")
    for clip in raw["clips"]:
        clip["annotation_complete"] = False
    wb.store.save_annotations("demo", raw)
    try:
        wb.preparation.request("demo")
        wb.preparation.start()
        wait_for(lambda: wb.preparation.status("demo")["ready"] == 2)
        assert len(calls) == 4
    finally:
        wb.preparation.close()
