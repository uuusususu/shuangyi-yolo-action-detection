"""整个工作台文件夹复制到别的电脑后，工程里的 YOLO 路径仍然能找到本机模型库里的文件。"""
from __future__ import annotations

from lstm_workbench import model_registry


def _library(tmp_path, monkeypatch):
    library = tmp_path / "models" / "yolo"
    library.mkdir(parents=True)
    (library / "a.onnx").write_bytes(b"x")
    monkeypatch.setattr(model_registry, "YOLO_LIBRARY", library)
    return library


def test_library_model_is_stored_by_file_name(tmp_path, monkeypatch):
    library = _library(tmp_path, monkeypatch)
    assert model_registry.portable_yolo_path(str(library / "a.onnx")) == "a.onnx"
    assert [item["path"] for item in model_registry.list_yolo_models()] == ["a.onnx"]


def test_relative_path_resolves_into_this_folders_library(tmp_path, monkeypatch):
    library = _library(tmp_path, monkeypatch)
    assert model_registry.resolve_yolo_path("a.onnx") == str(library / "a.onnx")


def test_old_absolute_path_from_another_pc_uses_local_library(tmp_path, monkeypatch):
    library = _library(tmp_path, monkeypatch)
    old = r"E:\别的电脑\lstm_workbench\models\yolo\a.onnx"
    assert model_registry.resolve_yolo_path(old) == str(library / "a.onnx")


def test_model_outside_library_is_kept_as_is(tmp_path, monkeypatch):
    _library(tmp_path, monkeypatch)
    outside = tmp_path / "elsewhere" / "b.onnx"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    assert model_registry.resolve_yolo_path(str(outside)) == str(outside)
    assert model_registry.portable_yolo_path(str(outside)) == str(outside)
