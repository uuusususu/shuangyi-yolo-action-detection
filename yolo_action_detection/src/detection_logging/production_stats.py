"""生产统计批次管理：计数、良率、归档和归零。"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ProductionBatch:
    """当前生产统计批次。"""
    started_at: float = 0.0
    total: int = 0
    ok: int = 0
    ng: int = 0

    @property
    def yield_rate(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.ok / self.total * 100.0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "total": self.total,
            "ok": self.ok,
            "ng": self.ng,
            "yield_rate": round(self.yield_rate, 2),
        }


@dataclass
class ArchiveResult:
    """归零操作结果。"""
    success: bool
    record_path: Optional[str] = None
    error: str = ""
    archived: bool = False  # 是否生成了记录文件


class ProductionStatsManager:
    """管理当前批次状态、记录归档和归零。

    - PASS/NG 调用 record_pass / record_ng 更新计数
    - 归零调用 reset_and_archive，有计数时先写 JSON 再清零
    - 检测启停不调用任何方法，统计不受影响
    """

    def __init__(
        self,
        records_dir: str | Path = "outputs/production_records",
    ) -> None:
        self._records_dir = Path(records_dir)
        self._batch = ProductionBatch(started_at=time.time())

    @property
    def batch(self) -> ProductionBatch:
        return self._batch

    @property
    def records_dir(self) -> Path:
        return self._records_dir

    def record_pass(self) -> None:
        self._batch.total += 1
        self._batch.ok += 1

    def record_ng(self) -> None:
        self._batch.total += 1
        self._batch.ng += 1

    def reset_and_archive(self) -> ArchiveResult:
        """归零并保存记录。

        - 有计数：先写 JSON，成功后清零，返回 archived=True
        - 空批次：不写文件，只刷新开始时间，返回 archived=False
        - 写入失败：保留当前数据，返回 success=False
        """
        if self._batch.total == 0:
            self._batch = ProductionBatch(started_at=time.time())
            return ArchiveResult(success=True, archived=False)

        ended_at = time.time()
        try:
            self._records_dir.mkdir(parents=True, exist_ok=True)
            record_path = self._write_record(ended_at)
        except Exception as exc:
            return ArchiveResult(success=False, error=str(exc), archived=False)

        self._batch = ProductionBatch(started_at=time.time())
        return ArchiveResult(success=True, record_path=str(record_path), archived=True)

    def _write_record(self, ended_at: float) -> Path:
        started_dt = datetime.fromtimestamp(self._batch.started_at)
        ended_dt = datetime.fromtimestamp(ended_at)
        filename = (
            f"record_{started_dt.strftime('%Y%m%d_%H%M%S')}"
            f"_{ended_dt.strftime('%Y%m%d_%H%M%S')}.json"
        )
        path = self._records_dir / filename
        record = {
            "schema_version": "production_record.v1",
            "started_at": self._batch.started_at,
            "ended_at": ended_at,
            "started_at_iso": started_dt.isoformat(timespec="seconds"),
            "ended_at_iso": ended_dt.isoformat(timespec="seconds"),
            "total": self._batch.total,
            "ok": self._batch.ok,
            "ng": self._batch.ng,
            "yield_rate": round(self._batch.yield_rate, 2),
        }
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def get_summary_text(self) -> str:
        return (
            f"开始时间: {datetime.fromtimestamp(self._batch.started_at).strftime('%Y-%m-%d %H:%M:%S')}  "
            f"总数: {self._batch.total}  OK: {self._batch.ok}  NG: {self._batch.ng}  "
            f"良率: {self._batch.yield_rate:.1f}%"
        )