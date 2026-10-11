"""开奖数据同步修复测试：重试诊断 + 手动录入。"""
import sqlite3
import sys
import tempfile
import os

sys.path.insert(0, "/home/hatch/workspace/ssq-analyzer")
from ssq_analyzer import SSQDataManager


def make_mgr():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return SSQDataManager(db_path=path), path


def test_manual_import_valid():
    mgr, path = make_mgr()
    ok = mgr.import_manual_draw("2026115", "2026-10-08", [3, 11, 18, 22, 29, 33], 9)
    assert ok is True
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT issue, date, r1, r2, r3, r4, r5, r6, blue FROM lottery_records WHERE issue='2026115'").fetchone()
    conn.close()
    assert row == ("2026115", "2026-10-08", 3, 11, 18, 22, 29, 33, 9), row
    os.unlink(path)
    print("PASS manual_import_valid")


def test_manual_import_idempotent():
    mgr, path = make_mgr()
    assert mgr.import_manual_draw("2026115", "2026-10-08", [3, 11, 18, 22, 29, 33], 9) is True
    # 重复录入不同号码：必须幂等跳过，不覆盖
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5, 6], 1) is True
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT r1, blue FROM lottery_records WHERE issue='2026115'").fetchone()
    conn.close()
    assert row == (3, 9), f"重复录入覆盖了原始数据: {row}"
    os.unlink(path)
    print("PASS manual_import_idempotent")


def test_manual_import_invalid():
    mgr, path = make_mgr()
    assert mgr.import_manual_draw("20261", "2026-10-08", [1, 2, 3, 4, 5, 6], 1) is False      # 期号位数错
    assert mgr.import_manual_draw("2026115", "10-08", [1, 2, 3, 4, 5, 6], 1) is False          # 日期格式错
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5], 1) is False        # 红球不足6个
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5, 5], 1) is False     # 红球重复
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5, 34], 1) is False    # 红球越界
    assert mgr.import_manual_draw("2026115", "2026-10-08", [1, 2, 3, 4, 5, 6], 17) is False    # 蓝球越界
    conn = sqlite3.connect(path)
    n = conn.execute("SELECT COUNT(*) FROM lottery_records").fetchone()[0]
    conn.close()
    assert n == 0, "非法录入不应写库"
    os.unlink(path)
    print("PASS manual_import_invalid")


def test_sync_failure_diagnostics():
    # 官方接口与备用源都失败时：同步返回 False，且 last_sync 有完整诊断信息
    import ssq_analyzer
    orig_get = ssq_analyzer.requests.get

    class FakeResp:
        status_code = 403
        text = "网宿云安全平台检测到您当前的访问行为存在异常，请稍后重试..."

    ssq_analyzer.requests.get = lambda *a, **k: FakeResp()
    try:
        mgr, path = make_mgr()
        ok = mgr.sync_official_data(fetch_count=2, retries=1)
        info = mgr.last_sync
        assert ok is False
        assert info["attempts"] == 1 and info["ok"] is False
        assert info["error"], "失败时必须记录诊断信息"
        assert "500" in info["error"], f"应注明备用源也失败: {info['error']}"
        print(f"PASS sync_failure_diagnostics (error={info['error'][:70]})")
        os.unlink(path)
    finally:
        ssq_analyzer.requests.get = orig_get


MOBILE_FIXTURE = """
<div class="kj-info-panel__title">
  <div class="flex">
    <span class="c111 fs-12 mr5">26116</span>
    <span class="fs-12 cB1B1B1">2026-10-11  </span>
  </div>
</div>
<div class="ball-box padding-b-13">
  <div class="hz-detail flex flex-wrap">
    <span class="red-ball ball">10</span>
    <span class="red-ball ball">19</span>
    <span class="red-ball ball">24</span>
    <span class="red-ball ball">27</span>
    <span class="red-ball ball">28</span>
    <span class="red-ball ball">33</span>
    <span class="blue-ball ball">12</span>
  </div>
</div>
"""

# 转义版 fixture（500 手机站有时返回 \\n \\' 这类转义 HTML）
MOBILE_FIXTURE_ESCAPED = MOBILE_FIXTURE.replace("\n", "\\n").replace("'", "\\'")

# 变体：期字在 span 内、日期后带星期
MOBILE_FIXTURE_V2 = """
<div class="kj-info-panel__title">
  <div class="flex">
    <span class="c111 fs-12 mr5">26116期</span>
    <span class="fs-12 cB1B1B1">2026-10-11 星期日 </span>
  </div>
</div>
<div class="ball-box padding-b-13">
  <div class="hz-detail flex flex-wrap">
    <span class="red-ball ball">10</span>
    <span class="red-ball ball">19</span>
    <span class="red-ball ball">24</span>
    <span class="red-ball ball">27</span>
    <span class="red-ball ball">28</span>
    <span class="red-ball ball">33</span>
    <span class="blue-ball ball">12</span>
  </div>
</div>
"""


def test_500_mobile_fallback():
    # 用本地 fixture 模拟 500 手机站，不依赖网络（覆盖原始与转义两种响应）
    import ssq_analyzer
    orig_get = ssq_analyzer.requests.get
    for fixture in (MOBILE_FIXTURE, MOBILE_FIXTURE_ESCAPED, MOBILE_FIXTURE_V2):
        class FakeResp:
            status_code = 200
            text = fixture

        ssq_analyzer.requests.get = lambda *a, **k: FakeResp()
        try:
            mgr, path = make_mgr()
            assert mgr._sync_from_500_mobile() is True
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT issue, date, r1, r2, r3, r4, r5, r6, blue FROM lottery_records"
                " WHERE issue='2026116'").fetchone()
            conn.close()
            assert row == ("2026116", "2026-10-11", 10, 19, 24, 27, 28, 33, 12), row
            os.unlink(path)
        finally:
            ssq_analyzer.requests.get = orig_get
    print("PASS 500_mobile_fallback")


def test_500_mobile_bad_html():
    import ssq_analyzer
    orig_get = ssq_analyzer.requests.get

    class FakeResp:
        status_code = 200
        text = "<html>no data here</html>"

    ssq_analyzer.requests.get = lambda *a, **k: FakeResp()
    try:
        mgr, path = make_mgr()
        assert mgr._sync_from_500_mobile() is False
        os.unlink(path)
    finally:
        ssq_analyzer.requests.get = orig_get
    print("PASS 500_mobile_bad_html")


if __name__ == "__main__":
    test_manual_import_valid()
    test_manual_import_idempotent()
    test_manual_import_invalid()
    test_sync_failure_diagnostics()
    test_500_mobile_fallback()
    test_500_mobile_bad_html()
    print("ALL 6 SYNC-FIX TESTS PASSED")
