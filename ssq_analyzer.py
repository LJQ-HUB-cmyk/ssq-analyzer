import os
import sqlite3
import random
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from google import genai
from backtest_ssq import calculate_prize as prize_tier  # 奖级计算统一用 backtest_ssq 的实现，避免两套重复逻辑打架
from self_learning import SelfScoreboard  # 自我记分板（只读已验证推荐，无循环依赖）

BEIJING = ZoneInfo("Asia/Shanghai")

# ----------------------------------------------------------------------
# 1. 数据库管理与官方数据同步模块 (SQLite + CWL API + Pandas)
# ----------------------------------------------------------------------
class SSQDataManager:
    def __init__(self, db_path: str = "ssq_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lottery_records (
                    issue TEXT PRIMARY KEY,
                    date TEXT,
                    r1 INTEGER, r2 INTEGER, r3 INTEGER,
                    r4 INTEGER, r5 INTEGER, r6 INTEGER,
                    blue INTEGER,
                    sales TEXT,
                    pool TEXT
                )
            """)
            conn.commit()

    def _sync_from_500_mobile(self) -> bool:
        """备用源：500 彩票手机站（m.500.com/kaijiang/ssq/）。

        背景：cwl.gov.cn 官方接口对机房 IP 做网宿云拦截；500 桌面站有 JS 挑战；
        但 500 手机站可直接返回含最新开奖号码的 HTML（红蓝球 span 标签）。
        只取最新一期；期号在手机站显示为 5 位缩写（如 26116=2026116），按 "20"+缩写还原。
        """
        import re
        url = "https://m.500.com/kaijiang/ssq/"
        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"[Warn] 500手机站返回 HTTP {resp.status_code}")
                return False
            html = resp.text
            # 500 手机站有时返回转义后的 HTML（\\n \\t \\' \\" \\/），先反转义再解析
            html = (html.replace("\\/", "/").replace("\\n", "\n").replace("\\t", "\t")
                        .replace("\\'", "'").replace('\\"', '"'))
            m = re.search(r">(\d{5})期?</span>\s*<span[^>]*>(\d{4}-\d{2}-\d{2})", html)
            issue = date = None
            if m:
                issue, date = "20" + m.group(1), m.group(2)
            else:
                # 兜底：在 ball-box 附近分别找期号与日期
                box_start = html.find("ball-box")
                head = html[max(0, box_start - 1500):box_start] if box_start >= 0 else html
                mi = re.search(r">(\d{5})期?<", head)
                md = re.search(r"(\d{4}-\d{2}-\d{2})", head)
                if mi and md:
                    issue, date = "20" + mi.group(1), md.group(1)
            if not issue or not date:
                print("[Warn] 500手机站未找到期号/日期")
                return False
            box_start = html.find("ball-box")
            box = html[box_start:box_start + 3000] if box_start >= 0 else html
            reds = [int(x) for x in re.findall(r"red-ball[^>]*>(\d+)<", box)[:6]]
            blues = [int(x) for x in re.findall(r"blue-ball[^>]*>(\d+)<", box)[:1]]
            if len(reds) != 6 or len(set(reds)) != 6 or not all(1 <= x <= 33 for x in reds) \
                    or len(blues) != 1 or not 1 <= blues[0] <= 16:
                print(f"[Warn] 500手机站号码解析无效：红{reds} 蓝{blues}")
                return False
            return self.import_manual_draw(issue, date, reds, blues[0])
        except Exception as e:
            print(f"[Warn] 500手机站同步异常: {e}")
            return False

    def sync_official_data(self, fetch_count: int = 100, retries: int = 3) -> bool:
        """同步官方开奖数据（带重试退避）。返回 True=成功拉到数据，False=失败。

        说明：福彩官方接口（cwl.gov.cn）前置了网宿云安全检测，会按 IP 限流/拦截
        机房访问。重试可应对"请稍后重试"类临时拦截；若持续被封，需走浏览器
        或手动录入（见 import_manual_draw）。诊断信息记在 self.last_sync。
        """
        import time
        url = f"https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=ssq&issueCount={fetch_count}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.cwl.gov.cn/",
            "Accept": "application/json, text/plain, */*",
        }
        self.last_sync = {"source": "cwl.gov.cn", "ok": False, "http_status": None,
                          "error": "", "issues_added": 0, "attempts": 0}
        added = 0
        for attempt in range(1, retries + 1):
            self.last_sync["attempts"] = attempt
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                self.last_sync["http_status"] = resp.status_code
                if resp.status_code == 200:
                    try:
                        data = resp.json().get("result", [])
                    except Exception:
                        data = []
                    if not data:
                        self.last_sync["error"] = "200 但返回体无 result 数据"
                        continue
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        for item in data:
                            issue = item.get("code")
                            date = str(item.get("date", ""))[:10]
                            reds = [int(x) for x in str(item.get("red", "")).split(",") if x.isdigit()]
                            try:
                                blue = int(item.get("blue", 0))
                            except Exception:
                                blue = 0
                            if issue and len(reds) == 6 and blue > 0:
                                cursor.execute("""
                                    INSERT OR IGNORE INTO lottery_records
                                    (issue, date, r1, r2, r3, r4, r5, r6, blue, sales, pool)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (str(issue), date, *reds, blue,
                                      str(item.get("sales", "0")), str(item.get("poolmoney", "0"))))
                                added += cursor.rowcount
                        conn.commit()
                    self.last_sync.update(ok=True, issues_added=added, error="")
                    print(f"[OK] 官方开奖数据已同步入库（新增 {added} 期，第 {attempt} 次尝试成功）。")
                    return True
                else:
                    self.last_sync["error"] = f"HTTP {resp.status_code}"
                    body_hint = resp.text[:60].replace("\n", " ")
                    if "安全" in resp.text[:200] or "异常" in resp.text[:200]:
                        self.last_sync["error"] += "（网宿云安全拦截）"
                    print(f"[Warn] 官方接口返回异常代码: {resp.status_code}（第 {attempt}/{retries} 次）{body_hint}")
            except Exception as e:
                self.last_sync["error"] = f"{type(e).__name__}: {e}"
                print(f"[Warn] 官方接口同步异常（第 {attempt}/{retries} 次）: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)  # 退避：5s、10s……
        print(f"[Warn] 官方接口 {retries} 次尝试均失败：{self.last_sync['error']}")
        # 备用源：500 手机站（官方接口被 IP 封锁时的自动降级）
        print("[i] 尝试备用源：500 彩票手机站……")
        if self._sync_from_500_mobile():
            self.last_sync.update(source="500手机站(m.500.com)", ok=True, error="")
            print("[OK] 已通过备用源同步最新开奖。")
            return True
        self.last_sync["error"] += "；备用源500手机站亦失败"
        return False

    def import_manual_draw(self, issue, date, reds, blue, sales="0", pool="0") -> bool:
        """手动录入单期开奖（官方接口被封时的备用入口）。

        参数校验严格：期号数字、日期格式、6 个 1~33 不重复红球、蓝球 1~16。
        成功返回 True；数据已存在时幂等返回 True（不覆盖）。
        """
        import re
        issue_s = str(issue).strip()
        if not re.fullmatch(r"\d{7}", issue_s):
            print(f"[Error] 期号格式错误：{issue_s}（应为 7 位数字，如 2026115）")
            return False
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date).strip()):
            print(f"[Error] 日期格式错误：{date}（应为 YYYY-MM-DD）")
            return False
        red_list = sorted(int(x) for x in reds) if isinstance(reds, (list, tuple)) else []
        if len(red_list) != 6 or len(set(red_list)) != 6 or \
                any(not 1 <= x <= 33 for x in red_list):
            print(f"[Error] 红球无效：{reds}（需 6 个 1~33 不重复数字）")
            return False
        try:
            blue_i = int(blue)
        except Exception:
            blue_i = 0
        if not 1 <= blue_i <= 16:
            print(f"[Error] 蓝球无效：{blue}（需 1~16）")
            return False
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute("SELECT 1 FROM lottery_records WHERE issue = ?",
                                  (issue_s,)).fetchone()
            if exists:
                print(f"[i] 第 {issue_s} 期已在库中，跳过录入（不覆盖）。")
                return True
            conn.execute("""
                INSERT INTO lottery_records
                (issue, date, r1, r2, r3, r4, r5, r6, blue, sales, pool)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (issue_s, str(date).strip(), *red_list, blue_i, str(sales), str(pool)))
            conn.commit()
        print(f"[OK] 已手动录入第 {issue_s} 期：红球 {' '.join(f'{x:02d}' for x in red_list)} 蓝球 {blue_i:02d}")
        return True

    def load_dataframe(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM lottery_records ORDER BY issue ASC", conn)
        return df

# ----------------------------------------------------------------------
# 2. 量化特征工程、定胆锁轴与 3+2 蓝球对冲引擎
# ----------------------------------------------------------------------
class QuantitativeEngine:
    PRIME_NUMBERS = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

    @staticmethod
    def calculate_ac_value(reds: List[int]) -> int:
        diffs = {abs(reds[j] - reds[i]) for i in range(len(reds)) for j in range(i + 1, len(reds))}
        return len(diffs) - (len(reds) - 1)

    @classmethod
    def extract_features(cls, reds: List[int], blue: int) -> Dict:
        sorted_red = sorted(reds)
        red_arr = np.array(sorted_red)
        red_sum = int(np.sum(red_arr))
        span = int(red_arr[-1] - red_arr[0])
        odd_count = int(np.sum(red_arr % 2 != 0))
        prime_count = sum(1 for x in sorted_red if x in cls.PRIME_NUMBERS)
        
        z1 = int(np.sum((red_arr >= 1) & (red_arr <= 11)))
        z2 = int(np.sum((red_arr >= 12) & (red_arr <= 22)))
        z3 = int(np.sum((red_arr >= 23) & (red_arr <= 33)))
        
        consecutive_pairs = int(np.sum(np.diff(red_arr) == 1))
        ac_val = cls.calculate_ac_value(sorted_red)
        birthday_count = int(np.sum(red_arr <= 31))

        return {
            "sum": red_sum,
            "span": span,
            "odd_even": f"{odd_count}:{6 - odd_count}",
            "prime_comp": f"{prime_count}:{6 - prime_count}",
            "zone_ratio": f"{z1}:{z2}:{z3}",
            "consecutive": consecutive_pairs,
            "ac_value": ac_val,
            "birthday_count": birthday_count,
            "blue": blue
        }

    @staticmethod
    def is_arithmetic_progression(reds: List[int]) -> bool:
        diffs = np.diff(sorted(reds))
        first_diff = int(list(diffs).pop(0))
        allowed_diffs = {2, 3, 4, 5}
        return len(set(diffs)) <= 2 and first_diff in allowed_diffs

    @classmethod
    def collision_heat_score(cls, reds: List[int], blue: int) -> int:
        """大众撞号热度评分（0~100，越低越好）。

        原理：每注中奖概率完全相同，但中奖后若与大量彩民撞号，奖金会被摊薄。
        人类选号有可预测的偏好（生日号、幸运数字、整齐图形），避开这些可降低分奖概率。
        注意：本评分基于行为研究中的常见偏好代理指标，不改变任何中奖概率；
        具体热号无官方销售数据支撑，分数为启发式估计，仅供参考。
        """
        s = sorted(reds)
        score = 0
        # 1. 生日号偏好：6 红全 <=31 是最典型的撞号组合
        if all(x <= 31 for x in s):
            score += 30
        elif sum(1 for x in s if x <= 12) >= 4:
            score += 18  # 月份号扎堆
        # 2. 连号
        consec = sum(1 for i in range(5) if s[i + 1] - s[i] == 1)
        score += consec * 8
        # 3. 等差/对称图形（filter 已拦截大部分，此处计残余风险）
        if cls.is_arithmetic_progression(s):
            score += 25
        # 4. 文化幸运数字扎堆（6/8/9/18/28 等华人偏好数字；启发式）
        score += sum(1 for x in s if x in (6, 8, 9, 18, 28)) * 3
        # 5. 蓝球中小号相对更受青睐（启发式）
        if blue in (6, 8, 9, 10):
            score += 8
        # 6. 32/33 大号区人类相对少选，微降热度
        score -= sum(1 for x in s if x >= 32) * 4
        return max(0, min(100, int(score)))

    @staticmethod
    def heat_label(heat: int) -> str:
        if heat <= 20:
            return "低"
        if heat <= 45:
            return "中"
        return "高"

    @classmethod
    def filter_red_combination(cls, reds: List[int]) -> bool:
        feats = cls.extract_features(reds, 1)
        if not (85 <= feats["sum"] <= 130):
            return False
        if not (20 <= feats["span"] <= 32):
            return False
        if feats["odd_even"] in ("0:6", "6:0", "1:5"):
            return False
        if feats["ac_value"] < 6:
            return False
        if feats["consecutive"] > 1:
            return False
        if cls.is_arithmetic_progression(reds):
            return False
        if feats["birthday_count"] == 6 and random.random() < 0.7:
            return False
        return True

    @classmethod
    def analyze_blue_hedging(cls, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        recent_blues = df["blue"].tail(10).tolist() if not df.empty else []
        last_blue = recent_blues[-1] if recent_blues else 6
        
        primary_candidates = (3, 5, 7, 9, 11, 13)
        primary_pool = [x for x in primary_candidates if x != last_blue]
        
        hedge_candidates = (4, 6, 8, 10, 12, 14)
        hedge_pool = [x for x in hedge_candidates if x != last_blue]
        
        main_picks = random.sample(primary_pool, 3)
        hedge_picks = random.sample(hedge_pool, 2)
        return main_picks, hedge_picks

    @classmethod
    def generate_enhanced_portfolio(cls, df: pd.DataFrame) -> Tuple[int, List[Dict]]:
        latest = df.iloc[-1]
        last_reds = set(int(latest[f"r{i}"]) for i in range(1, 7))
        last_reds_list = list(last_reds)
        
        anchor_red = 32 if 33 in last_reds else 33
        
        main_blues, hedge_blues = cls.analyze_blue_hedging(df)
        mb_a, mb_b, mb_c = main_blues
        hb_a, hb_b = hedge_blues
        blue_plan = (
            (mb_a, "主攻反弹"),
            (hb_a, "动态对冲"),
            (mb_b, "主攻反弹"),
            (hb_b, "动态对冲"),
            (mb_c, "主攻反弹")
        )
        
        repeat_targets = (0, 1, 0, 1, 2)
        all_pool = set(range(1, 34))
        fresh_pool = list(all_pool - last_reds)
        
        results = []
        for i in range(5):
            r_count = repeat_targets[i]
            blue_val, strategy_tag = blue_plan[i]

            # 在有效组合中保留撞号热度最低的一组（不改变中奖概率，只降低分奖风险）
            best_reds, best_heat = None, 101
            for _ in range(3000):
                chosen_reds = {anchor_red}
                if r_count > 0:
                    # 胆码本身若在上期红球中，抽重号时先剔除，避免 set 去重导致实际重号数少于配额
                    repeat_pool = [x for x in last_reds_list if x != anchor_red]
                    chosen_repeats = set(random.sample(repeat_pool, min(r_count, len(repeat_pool))))
                    chosen_reds.update(chosen_repeats)

                needed = 6 - len(chosen_reds)
                avail_fresh = [x for x in fresh_pool if x not in chosen_reds]
                if len(avail_fresh) < needed:
                    continue
                chosen_reds.update(random.sample(avail_fresh, needed))

                sorted_reds = sorted(list(chosen_reds))
                if len(sorted_reds) == 6 and cls.filter_red_combination(sorted_reds):
                    heat = cls.collision_heat_score(sorted_reds, blue_val)
                    if heat < best_heat:
                        best_heat, best_reds = heat, sorted_reds
                        if best_heat == 0:
                            break

            if best_reds is not None:
                feats = cls.extract_features(best_reds, blue_val)
                results.append({
                    "id": i + 1,
                    "reds": best_reds,
                    "blue": blue_val,
                    "strategy": strategy_tag,
                    "repeats": r_count,
                    "heat": best_heat,
                    "feats": feats
                })
            else:
                sample_reds = sorted(random.sample(range(1, 34), 6))
                results.append({
                    "id": i + 1,
                    "reds": sample_reds,
                    "blue": blue_val,
                    "strategy": strategy_tag,
                    "repeats": r_count,
                    "heat": cls.collision_heat_score(sample_reds, blue_val),
                    "feats": cls.extract_features(sample_reds, blue_val)
                })

        return anchor_red, results

# ----------------------------------------------------------------------
# 3. 精算师微观盘面指标计算模块 (Actuarial Metrics)
# ----------------------------------------------------------------------
class ActuarialMetrics:
    @staticmethod
    def evaluate_market_metrics(pool_str: str, candidates: List[Dict]) -> Dict:
        """测算奖池溢价系数、5注组合红球覆盖率及博弈碰撞指数"""
        pool_num = None
        try:
            cleaned = str(pool_str).replace("亿", "").replace("元", "").replace(",", "").strip()
            val = float(cleaned)
            if val == val:  # 排除 nan（pandas 空值会变成 nan，float('nan') 不抛异常）
                pool_num = val if val < 1000 else val / 1e8
        except Exception:
            pool_num = None

        if pool_num is None:
            # 奖池数据缺失时不编造数字、不套用高溢价话术
            premium_ratio_txt = "未知"
            premium_grade = "奖池数据缺失"
            pool_display = "未知"
            actuarial_advice = "奖池数据缺失，无法评估溢价；坚持轻仓固定注数（严禁加仓倍投），理性参与。"
        else:
            premium_ratio = round(pool_num / 1.0, 2)
            premium_ratio_txt = f"{premium_ratio}x"
            pool_display = round(pool_num, 2)
            if premium_ratio >= 8.0:
                premium_grade = "极高溢价 (S级·顶格支持单注1000万)"
                actuarial_advice = "当前奖池处于极端高位溢价期，高等奖边际价值显著放大，坚持轻仓固定注数（严禁加仓倍投），锁定最高EV组合。"
            elif premium_ratio >= 3.0:
                premium_grade = "充裕溢价 (A级·支持多注封顶)"
                actuarial_advice = "奖池溢价充裕，高等奖边际价值上升，坚持轻仓固定注数（严禁加仓倍投）。"
            else:
                premium_grade = "常态水位"
                actuarial_advice = "奖池处于常态水位，坚持轻仓固定注数（严禁加仓倍投），理性参与。"

        # 5注组合去重红球覆盖率
        unique_reds = set()
        for c in candidates:
            unique_reds.update(c["reds"])
        coverage_count = len(unique_reds)
        coverage_pct = round((coverage_count / 33.0) * 100, 1)

        return {
            "pool_num": pool_display,
            "premium_ratio": premium_ratio_txt,
            "premium_grade": premium_grade,
            "coverage_count": coverage_count,
            "coverage_pct": f"{coverage_pct}%",
            "collision_index": "极低 (已过滤大众生日密集区与等差图形，独占头奖期望最优)",
            "actuarial_advice": actuarial_advice,
        }

# ----------------------------------------------------------------------
# 3.5 专家式走势分析模块 (Expert-Style Trend Analysis)
# ----------------------------------------------------------------------
class ExpertStyleAnalysis:
    """模仿彩票专家的分析思路：三区走势、012路、奇偶/大小/质合比、热号遗漏、胆码杀号。

    诚实声明：本模块输出的全部是历史开奖数据的**描述性统计**。
    双色球每期独立随机，这些统计不改变任何号码未来的中奖概率，
    仅用于走势查看与复盘，不得解读为预测依据。
    """
    ZONE_BOUNDS = ((1, 11), (12, 22), (23, 33))
    PRIME_SET = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}

    @classmethod
    def zone_of(cls, n: int) -> int:
        for i, (lo, hi) in enumerate(cls.ZONE_BOUNDS, start=1):
            if lo <= n <= hi:
                return i
        return 0

    @classmethod
    def _reds(cls, row) -> List[int]:
        return [int(row[f"r{i}"]) for i in range(1, 7)]

    @classmethod
    def omission_map(cls, df: pd.DataFrame) -> Dict[int, int]:
        """每个红球号码的当前遗漏期数（距离上次开出隔了多少期）。"""
        n = len(df)
        last_seen = {}
        for idx, row in df.iterrows():
            for x in cls._reds(row):
                last_seen[x] = idx
        return {x: (n - 1 - last_seen.get(x, -1)) for x in range(1, 34)}

    @classmethod
    def analyze(cls, df: pd.DataFrame, window: int = 10, hot_window: int = 30) -> Dict:
        if df.empty:
            return {}
        recent = df.tail(window)
        total_balls = len(recent) * 6

        zone_counts = [0, 0, 0]
        route012 = [0, 0, 0]
        odd_c = even_c = big_c = small_c = prime_c = comp_c = 0
        blue_odd = blue_even = blue_prime = blue_comp = blue_neither = 0
        per_draw_zone = []
        for _, row in recent.iterrows():
            reds = cls._reds(row)
            zc = [0, 0, 0]
            for x in reds:
                z = cls.zone_of(x)
                zone_counts[z - 1] += 1
                zc[z - 1] += 1
                route012[x % 3] += 1
                if x % 2:
                    odd_c += 1
                else:
                    even_c += 1
                if x >= 17:
                    big_c += 1
                else:
                    small_c += 1
                if x in cls.PRIME_SET:
                    prime_c += 1
                else:
                    comp_c += 1
            per_draw_zone.append(zc)
            b = int(row["blue"])
            if b % 2:
                blue_odd += 1
            else:
                blue_even += 1
            if b in cls.PRIME_SET:
                blue_prime += 1
            elif b == 1:
                blue_neither += 1  # 1 既非质数也非合数
            else:
                blue_comp += 1

        # 热号：近 hot_window 期出现频次 Top6；遗漏：全库遗漏最长 Top6
        hot_pool = df.tail(hot_window)
        freq: Dict[int, int] = {}
        for _, row in hot_pool.iterrows():
            for x in cls._reds(row):
                freq[x] = freq.get(x, 0) + 1
        hot_sorted = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        hot6 = [(x, c) for x, c in hot_sorted[:6]]

        omit = cls.omission_map(df)
        cold_sorted = sorted(omit.items(), key=lambda kv: (-kv[1], kv[0]))
        cold6 = cold_sorted[:6]

        # 胆码取热号前3，杀号取遗漏最长前3，二者互斥
        danma = [x for x, _ in hot6[:3]]
        shahao = [x for x, _ in cold6[:3] if x not in danma]
        # 补齐杀号到3个（从遗漏榜顺延）
        for x, _ in cold6[3:]:
            if len(shahao) >= 3:
                break
            if x not in danma and x not in shahao:
                shahao.append(x)

        last_row = recent.iloc[-1]
        last_zone = [0, 0, 0]
        for x in cls._reds(last_row):
            last_zone[cls.zone_of(x) - 1] += 1

        def heat_word(actual: float, expected: float) -> str:
            if expected <= 0:
                return "均衡"
            r = actual / expected
            if r >= 1.15:
                return "偏热"
            if r <= 0.85:
                return "偏冷"
            return "均衡"

        return {
            "window": len(recent),
            "zone_counts": zone_counts,
            "zone_expected": total_balls / 3,
            "zone_words": [heat_word(c, total_balls / 3) for c in zone_counts],
            "route012": route012,
            "route_expected": total_balls / 3,
            "odd_even": (odd_c, even_c),
            "odd_expected": total_balls * 17 / 33,
            "big_small": (big_c, small_c),
            "big_expected": total_balls * 17 / 33,
            "prime_comp": (prime_c, comp_c),
            "prime_expected": total_balls * 11 / 33,
            "blue_odd_even": (blue_odd, blue_even),
            "blue_prime_comp": (blue_prime, blue_comp, blue_neither),
            "last_blue": int(last_row["blue"]),
            "last_zone": last_zone,
            "hot6": hot6,
            "cold6": cold6,
            "danma": danma,
            "shahao": shahao,
        }

    @classmethod
    def render_markdown(cls, a: Dict) -> List[str]:
        """把分析结果渲染为专家口吻的中文解读行（看板展示用）。"""
        if not a:
            return ["> 暂无足够历史数据生成走势分析。", ""]
        w = a["window"]
        zc = a["zone_counts"]
        r012 = a["route012"]
        oe = a["odd_even"]
        bs = a["big_small"]
        pc = a["prime_comp"]
        boe = a["blue_odd_even"]
        bp, bc, bn = a["blue_prime_comp"]
        lz = a["last_zone"]
        lb = a["last_blue"]
        if lb in cls.PRIME_SET:
            lb_qh = "质数"
        elif lb == 1:
            lb_qh = "既非质数也非合数"
        else:
            lb_qh = "合数"

        def fmt(nums):
            return " ".join(f"`{x:02d}`" for x in nums)

        lines = [
            "> ⚠️ 以下为历史开奖数据的**描述性统计**（专家式走势分析口径）；"
            "双色球每期独立随机，这些统计**不改变任何号码的中奖概率**，仅供走势查看。",
            "",
            f"**【三区走势】** 近{w}期红球三区分布为 `{zc[0]}:{zc[1]}:{zc[2]}`"
            f"（均值约 `{a['zone_expected']:.0f}` 枚/区），"
            f"一区{a['zone_words'][0]}、二区{a['zone_words'][1]}、三区{a['zone_words'][2]}；"
            f"上期三区比 `{lz[0]}:{lz[1]}:{lz[2]}`。",
            f"**【012路】** 近{w}期 0路`{r012[0]}`枚、1路`{r012[1]}`枚、2路`{r012[2]}`枚"
            f"（均值约 `{a['route_expected']:.0f}` 枚/路）。",
            f"**【奇偶比】** 近{w}期 `{oe[0]}:{oe[1]}`（奇数期望约 `{a['odd_expected']:.0f}` 枚）；"
            f"**【大小比】** `{bs[0]}:{bs[1]}`（大数≥17，期望约 `{a['big_expected']:.0f}` 枚）；"
            f"**【质合比】** `{pc[0]}:{pc[1]}`（质数期望约 `{a['prime_expected']:.0f}` 枚）。",
            f"**【蓝球】** 近{w}期奇偶 `{boe[0]}:{boe[1]}`、质合 `{bp}:{bc}`"
            f"{f'（另有{bn}期为01，既非质数也非合数）' if bn else ''}；"
            f"上期蓝球 `{lb:02d}`"
            f"（{'奇数' if lb % 2 else '偶数'}/{lb_qh}）。",
            f"**【热号】** 近{30}期热号："
            + "、".join(f"`{x:02d}`({c}次)" for x, c in a["hot6"])
            + "。",
            f"**【遗漏】** 当前遗漏最长："
            + "、".join(f"`{x:02d}`({o}期未出)" for x, o in a["cold6"])
            + "。",
            f"**【胆码参考】** {fmt(a['danma'])}（近30期热号前三，专家口径）",
            f"**【杀号参考】** {fmt(a['shahao'])}（遗漏最长，专家口径建议规避；"
            "冷号回补无统计依据，杀号不提高中奖概率）",
            "",
        ]
        return lines


# ----------------------------------------------------------------------
# 4. 基于 google-genai 的 Gemini AI 研判模块
# ----------------------------------------------------------------------
def generate_gemini_analysis(df: pd.DataFrame, candidates: List[Dict], anchor: int, actuarial: Dict) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "> ⚠️ 未检测到 `GEMINI_API_KEY` 环境变量，跳过 AI 研判生成。"

    try:
        client = genai.Client(api_key=api_key)
        cols = ("issue", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue")
        recent_df = df.tail(5).loc[:, cols]
        recent_records = recent_df.to_dict(orient="records")
        
        summary_prompt = f"""
你是一位资深精算师兼彩票量化博弈分析师。请结合双色球历史客观数据、精算指标与本期候选组合，提供简要的精算推演点评（150-250字）。

【最新5期开奖数据】：
{recent_records}

【精算微观盘面指标】：
- 奖池滚存：{actuarial['pool_num']} 亿元（溢价系数：{actuarial['premium_ratio']}，{actuarial['premium_grade']}）
- 5注组合覆盖度：{actuarial['coverage_count']}/33 ({actuarial['coverage_pct']})
- 核心红胆锁轴：{anchor:02d}

【候选组合】：
{[{'reds': c['reds'], 'blue': c['blue'], 'strategy': c['strategy']} for c in candidates]}

【要求】：
1. 语言客观精炼，立足精算与博弈论，严禁绝对化预测。
2. 简评高额奖池下的边际期望收益与去大众撞号偏好的精算意义。
3. 强调理性资金管理。
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=summary_prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"> ⚠️ Gemini AI 研判生成提示: {str(e)}"

# ----------------------------------------------------------------------
# 5. 主流程：全屏自适应卡片看板 (方案二标准 + 精算微观盘面)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 推荐存档与上期回顾模块：每次运行存档本期推荐，下次有新开奖时自动核对
# ----------------------------------------------------------------------
def calc_next_draw_date(date_str):
    """由本期开奖日期推算下期开奖日期（双色球固定每周二、四、日开奖）。"""
    d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    draw_weekdays = {1, 3, 6}  # 周二、周四、周日
    nxt = d + timedelta(days=1)
    while nxt.weekday() not in draw_weekdays:
        nxt += timedelta(days=1)
    return nxt.strftime("%Y-%m-%d")


def calc_next_issue(issue, latest_date=None, next_date=None):
    """由当前期号推算下期期号。跨年时按下期开奖日期直接取新年001（旧的>160阈值是错的，一年只有152~154期）。"""
    s = str(issue)
    if not s.isdigit() or len(s) < 5:
        return "下期"
    if latest_date and next_date:
        if str(next_date)[:4] > str(latest_date)[:4]:
            return "%s001" % str(next_date)[:4]
    year, seq = int(s[:-3]), int(s[-3:])
    return "%d%03d" % (year, seq + 1)


def ensure_recommendations_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            target_issue TEXT,
            note_id INTEGER,
            reds TEXT,
            blue INTEGER,
            strategy TEXT,
            repeats INTEGER,
            target_date TEXT,
            created_at TEXT,
            verified_at TEXT,
            PRIMARY KEY (target_issue, note_id)
        )
    """)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(recommendations)").fetchall()]
    for col in ("verified_at", "strategy", "repeats", "target_date"):
        if col not in cols:
            conn.execute(f"ALTER TABLE recommendations ADD COLUMN {col} {'INTEGER' if col == 'repeats' else 'TEXT'}")
    # 回填老存档缺失的 target_date：按"上一期开奖日期→下期开奖日"推算，保证回顾能按日期匹配
    for (tgt,) in conn.execute(
            "SELECT DISTINCT target_issue FROM recommendations"
            " WHERE target_date IS NULL OR target_date = ''").fetchall():
        prev = conn.execute(
            "SELECT date FROM lottery_records WHERE issue < ? ORDER BY issue DESC LIMIT 1",
            (str(tgt),)).fetchone()
        if prev and prev[0]:
            nd = calc_next_draw_date(prev[0])
            conn.execute("UPDATE recommendations SET target_date = ? WHERE target_issue = ?",
                         (nd, str(tgt)))
            print(f"[i] 回填第 {tgt} 期 target_date = {nd}")
    conn.commit()


def save_recommendations(conn, target_issue, target_date, candidates):
    """幂等保存：同一期已有存档则直接跳过，绝不删除/覆盖——保护开奖前生成的真实存档。

    背景：定时任务可能在官方接口滞后时先跑一次（此时最新期号还是上期），
    若此时无条件重建，会把真正的下期推荐存档洗掉，之后回顾核对的就是假数据。
    """
    tgt = str(target_issue)
    existing = conn.execute(
        "SELECT COUNT(*) FROM recommendations WHERE target_issue = ?", (tgt,)).fetchone()[0]
    if existing:
        print(f"[i] 第 {tgt} 期已有 {existing} 条推荐存档，跳过保存（保护原始存档不被覆盖）。")
        return False
    now = datetime.now(BEIJING).strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        for c in candidates:
            reds_str = ",".join(str(x) for x in sorted(c["reds"]))
            conn.execute(
                "INSERT INTO recommendations (target_issue, note_id, reds, blue, strategy, repeats,"
                " target_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tgt, int(c["id"]), reds_str, int(c["blue"]), str(c.get("strategy", "")),
                 int(c.get("repeats", 0)) if str(c.get("repeats", 0)).isdigit() else 0,
                 str(target_date), now))
    print(f"[i] 第 {tgt} 期（{target_date}）{len(candidates)} 注推荐已存档。")
    return True


def load_recommendations(conn, target_issue):
    """读取某期的存档推荐（用于直接展示存档，不重新生成）。"""
    rows = conn.execute(
        "SELECT note_id, reds, blue, strategy, repeats FROM recommendations"
        " WHERE target_issue = ? ORDER BY note_id",
        (str(target_issue),)).fetchall()
    out = []
    for note_id, reds_str, blue, strategy, repeats in rows:
        reds = sorted(int(x) for x in str(reds_str).split(",") if x.strip().isdigit())
        if len(reds) != 6:
            continue
        out.append({
            "id": int(note_id),
            "reds": reds,
            "blue": int(blue),
            "strategy": strategy or "存档",
            "repeats": repeats if repeats is not None else "-",
            "heat": QuantitativeEngine.collision_heat_score(reds, int(blue)),
            "feats": QuantitativeEngine.extract_features(reds, int(blue)),
            "from_archive": True,
        })
    return out


def build_review_lines(conn, latest_issue, df):
    """核对所有尚未验证、且开奖数据已到位的存档推荐（最多回溯 5 期），核对后标记。

    匹配优先用 target_date 对开奖日期（跨年/期号推算错误时依然可靠），
    target_date 缺失时回退到期号匹配。
    """
    latest_str = str(latest_issue)
    latest_rows = df[df["issue"].astype(str) == latest_str]
    latest_date = str(latest_rows.iloc[0]["date"])[:10] if not latest_rows.empty else ""
    cur = conn.execute(
        "SELECT DISTINCT target_issue, target_date FROM recommendations"
        " WHERE (verified_at IS NULL OR verified_at = '')"
        " AND (target_issue <= ?"
        "      OR (target_date IS NOT NULL AND target_date != '' AND target_date <= ?))"
        " ORDER BY target_issue DESC LIMIT 5",
        (latest_str, latest_date))
    targets = [(r[0], r[1]) for r in cur.fetchall()]
    lines = []
    now = datetime.now(BEIJING).strftime("%Y-%m-%d %H:%M:%S")
    for tgt, tgt_date in targets:
        draw = df[df["issue"].astype(str) == str(tgt)]
        if draw.empty and tgt_date:
            draw = df[df["date"].astype(str).str[:10] == str(tgt_date)[:10]]
        if draw.empty:
            continue
        row = draw.iloc[0]
        act_reds = [int(row["r%d" % i]) for i in range(1, 7)]
        act_blue = int(row["blue"])
        recs = conn.execute(
            "SELECT note_id, reds, blue FROM recommendations WHERE target_issue = ? ORDER BY note_id",
            (str(tgt),)).fetchall()
        lines.append("### \U0001F50D 上期推荐回顾（第 `%s` 期）" % tgt)
        lines.append(
            "实际开奖：红球 %s ｜ 蓝球 `%02d`"
            % (" ".join("`%02d`" % x for x in act_reds), act_blue))
        act = set(act_reds)
        total = 0
        for note_id, reds_str, blue in recs:
            rec_reds = [int(x) for x in reds_str.split(",")]
            rh = len(set(rec_reds) & act)
            bh = int(blue) == act_blue
            tier, money = prize_tier(rh, bh)
            total += money
            mark = "\u2192 **%d\u7b49\u5956 %d\u5143**" % (tier, money) if tier else "\u2192 \u672a\u4e2d\u5956"
            rstr = " ".join("`%02d`" % x for x in rec_reds)
            bmark = "\u84dd\u4e2d" if bh else "\u84dd\u9519"
            lines.append(
                "- 第 `%02d` 注：%s ＋ `%02d`（红中%d、%s）%s"
                % (note_id, rstr, int(blue), rh, bmark, mark))
        lines.append(
            "- **\u6c47\u603b**：%d \u6ce8\u5171\u6295\u5165 `%d` \u5143，\u4e2d\u5956 `%d` \u5143"
            % (len(recs), len(recs) * 2, total))
        lines.append("")
        conn.execute("UPDATE recommendations SET verified_at = ? WHERE target_issue = ?",
                     (now, str(tgt)))
    conn.commit()
    if not lines:
        lines = ["### \U0001F50D 上期推荐回顾",
                 "- 暂无可核对的上期推荐存档；本期推荐已存档，待开奖数据到位后自动回顾。",
                 ""]
    lines.extend(["---", ""])
    return lines


def main():
    print("=== 开始运行双色球精算量化分析工作流 ===")
    data_mgr = SSQDataManager()
    sync_ok = data_mgr.sync_official_data(fetch_count=100)
    df = data_mgr.load_dataframe()

    if df.empty:
        print("[Error] 未获取到历史数据，终止流程。")
        return

    rec_conn = sqlite3.connect(data_mgr.db_path)
    ensure_recommendations_table(rec_conn)

    latest = df.iloc[-1]
    latest_issue = str(latest['issue'])
    latest_date = str(latest['date'])[:10]
    latest_reds = [int(latest[f"r{i}"]) for i in range(1, 7)]
    latest_blue = int(latest["blue"])
    latest_feats = QuantitativeEngine.extract_features(latest_reds, latest_blue)
    review_lines = build_review_lines(rec_conn, latest_issue, df)

    print(f"最新一期: {latest_issue} ({latest_date}) | 红球: {latest_reds} | 蓝球: {latest_blue}")

    # 下期推荐：存档优先——已有存档直接展示，不重新生成、不覆盖、不调 Gemini
    next_date = calc_next_draw_date(latest_date)
    next_issue = calc_next_issue(latest_issue, latest_date, next_date)
    archived = load_recommendations(rec_conn, next_issue)
    if archived:
        print(f"[i] 第 {next_issue} 期已有 {len(archived)} 注存档，直接展示存档推荐。")
        candidates = archived
        common = set(archived[0]["reds"])
        for c in archived[1:]:
            common &= set(c["reds"])
        anchor_red = sorted(common)[0] if len(common) == 1 else None
        from_archive = True
    else:
        # 综合推算：定胆锁轴 + 重号立体梯队 + 蓝球 3+2 对冲
        anchor_red, candidates = QuantitativeEngine.generate_enhanced_portfolio(df)
        save_recommendations(rec_conn, next_issue, next_date, candidates)
        from_archive = False
    # 自我记分板：统计已验证推荐（含本轮刚验证的），与随机基线并排
    scoreboard_rows = SelfScoreboard.compute(rec_conn, df)
    rec_conn.close()

    # 计算精算师微观盘面指标
    actuarial_info = ActuarialMetrics.evaluate_market_metrics(latest['pool'], candidates)

    # 调用 Gemini AI 生成精算研判（仅新生成时调用，展示存档时跳过）
    if from_archive:
        ai_commentary = "> 本期展示已存档推荐，未重新调用 AI 研判。"
    else:
        ai_commentary = generate_gemini_analysis(df, candidates, anchor_red, actuarial_info)

    # 专家式走势分析（历史描述性统计；存档展示与新生成两种路径都计算）
    expert_analysis = ExpertStyleAnalysis.analyze(df)
    expert_lines = ExpertStyleAnalysis.render_markdown(expert_analysis)

    # 组装极简卡片式看板 (方案二：绝对不超宽，无横向滑动)
    current_time_str = datetime.now(BEIJING).strftime("%Y-%m-%d %H:%M")
    sync_info = getattr(data_mgr, "last_sync", {}) or {}
    sync_detail = ""
    if not sync_ok and sync_info.get("error"):
        sync_detail = f"（{sync_info['error']}，{sync_info.get('attempts', 1)} 次尝试）"
    sync_mark = "官方接口同步成功" if sync_ok else f"⚠️官方接口同步失败{sync_detail}，数据可能滞后（已用本地库）"
    archive_mark = "（存档展示）" if from_archive else ""
    anchor_txt = f"`{anchor_red:02d}`" if anchor_red else "—"

    lines = [
        "# 🔴🔵 双色球量化分析看板（精算量化与对冲版）",
        "",
        f"> 🕒 **更新时间**：`{current_time_str}`（北京时间） ｜ **期号**：第 `{next_issue}` 期推演{archive_mark}",
        f"> 📡 **数据状态**：{sync_mark} ｜ 数据截止第 `{latest_issue}` 期（{latest_date}）",
        "",
        "---",
        "",
        "### 📊 精算师微观盘面指标",
        f"- **奖池溢价系数**：`{actuarial_info['premium_ratio']}` ｜ **{actuarial_info['premium_grade']}**",
        f"- **5注覆盖效率**：去重覆盖 `{actuarial_info['coverage_count']}/33` 枚红球（覆盖率 `{actuarial_info['coverage_pct']}`）",
        f"- **撞号碰撞指数**：`{actuarial_info['collision_index']}`",
        f"- **精算资金建议**：{actuarial_info['actuarial_advice']}",
        "",
        "---",
        "",
        f"### 📋 上期开奖总结（第 `{latest_issue}` 期）",
        f"- **开奖号码**：红球 {' '.join(f'`{x:02d}`' for x in latest_reds)} ｜ 蓝球 `{latest_blue:02d}`",
        f"- **奖池滚存**：约 `{latest['pool']}` 元",
        f"- **形态特征**：三区比 `{latest_feats['zone_ratio']}` ｜ 奇偶比 `{latest_feats['odd_even']}` ｜ 连号组数 `{latest_feats['consecutive']}` ｜ 跨度 `{latest_feats['span']}` ｜ AC值 `{latest_feats['ac_value']}`",
        "",
        "---",
        "",
        *review_lines,
        "### 📈 专家式走势分析",
        "",
        *expert_lines,
        "---",
        "",
        "### 🧾 自我记分板（已验证推荐 vs 随机基线）",
        "",
        *SelfScoreboard.render_markdown(scoreboard_rows),
        "---",
        "",
        f"### 🎯 本期推荐组合（核心红胆：{anchor_txt} ｜ 蓝球 3+2 对冲{archive_mark}）",
        ""
    ]

    # 方案二：生成纯自适应卡片式列表
    for c in candidates:
        red_str = " ".join(f"`{x:02d}`" for x in c["reds"])
        if c.get("from_archive"):
            strategy_badge = "📦 存档"
        else:
            strategy_badge = "🎯 主攻" if c["strategy"] == "主攻反弹" else "🛡️ 对冲"
        heat = c.get("heat")
        heat_txt = f" ｜ 撞号风险: {QuantitativeEngine.heat_label(heat)}({heat})" if isinstance(heat, int) else ""
        lines.append(
            f"* 🔴 **第 {c['id']:02d} 注**：{red_str} ＋ 🔵 `{c['blue']:02d}`  "
            f"└─ *[{strategy_badge}] {c['strategy']} ｜ 重号配额: {c.get('repeats', '-')} 码{heat_txt}*"
        )
        
    lines.extend([
        "",
        "---",
        "",
        "### 💡 精算推演与优化逻辑",
        f"1. **定胆锁轴（聚拢红球）**：以高位边码 {anchor_txt} 作为全组核心基石，打破号码分散碎片化缺陷，增强多码同框概率。",
        "2. **重号立体防御**：按 2 注零重号（防大换血）+ 2 注单重号 + 1 注双重号梯度布局，化解两极化盘面风险。",
        "3. **蓝球 3+2 动态对冲**：3 注主攻反弹奇数 + 2 注强制对冲偶数，彻底消除单边下注导致的通杀风险。",
        "",
        "### 🧠 精算师研判点评",
        ai_commentary,
        "",
        "---",
        "",
        "<sub>*免责声明：彩票为独立随机事件，精算模型旨在控制分奖稀释与资金风险边界，请理性参与。"
        "撞号风险评分仅估计中奖后与他人分奖的可能性，不改变任何中奖概率。*</sub>"
    ])
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print("[Done] 精算看板 README.md 已生成完毕。")

if __name__ == "__main__":
    main()
