"""
포트폴리오 엔진 v3 — 정량 매크로 점수 + 레짐별 방어 비중.

설계 원칙(피드백 반영):
- 수동 매크로 가정("고금리/고유가/매파")을 코드에 박지 않는다. 시장 데이터에서 매번 점수를 계산한다.
- 4개 점수: risk_off, rates_up, inflation, fx_stress (모두 z-score).
- 시장 상태를 normal / caution / riskoff 로 분류하고, 레짐별로 성장자산 상한·방어자산 하한을 강제한다.
- ML/예측은 production 비중에 쓰지 않는다(모멘텀은 약한 틸트로만).

데이터 fetch( fetch_market_data )와 비중 계산( compute_weights )을 분리 → 계산부는 네트워크 없이 테스트 가능.
"""
import math


def _clip(x, lo=-2.0, hi=2.0):
    return max(lo, min(hi, float(x)))


def _zscore_latest(series):
    """리스트/시퀀스의 마지막 값을 과거 분포로 표준화."""
    xs = [v for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(xs) < 20:
        return 0.0
    mu = sum(xs) / len(xs)
    var = sum((v - mu) ** 2 for v in xs) / (len(xs) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return 0.0
    return (xs[-1] - mu) / sd


def classify_regime(scores, cfg):
    """방어 압력 종합 점수로 레짐 분류."""
    rc = scores["risk_off"] + 0.5 * max(scores["rates_up"], 0.0) + 0.5 * max(scores["fx_stress"], 0.0)
    th = cfg["regime"]
    if rc >= th["caution_to_riskoff"]:
        return "riskoff"
    if rc >= th["normal_to_caution"]:
        return "caution"
    return "normal"


GROWTH_GROUPS = {"growth", "income"}
DEF_GROUPS = {"gold", "longbond", "shortbond", "cash"}


def _normalize(w):
    s = sum(w.values())
    if s <= 0:
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: v / s for k, v in w.items()}


def _apply_regime_caps(w, uni, cfg, regime):
    w = _normalize(w)
    maxg = cfg["regime"]["max_growth_weight"][regime]
    mind = cfg["regime"]["min_defensive_weight"][regime]
    # 1) 성장자산 상한
    g = sum(w[k] for k in w if uni[k]["group"] in GROWTH_GROUPS)
    if g > maxg and g > 0:
        scale = maxg / g
        freed = 0.0
        for k in w:
            if uni[k]["group"] in GROWTH_GROUPS:
                freed += w[k] * (1 - scale)
                w[k] *= scale
        sb = [k for k in w if uni[k]["group"] in ("shortbond", "cash")]
        for k in sb:
            w[k] += freed / len(sb)
    # 2) 방어자산 하한
    d = sum(w[k] for k in w if uni[k]["group"] in DEF_GROUPS)
    if d < mind:
        need = mind - d
        g = sum(w[k] for k in w if uni[k]["group"] in GROWTH_GROUPS)
        if g > 0:
            scale = max(0.0, (g - need) / g)
            for k in w:
                if uni[k]["group"] in GROWTH_GROUPS:
                    w[k] *= scale
        sb = [k for k in w if uni[k]["group"] in ("shortbond", "cash")]
        for k in sb:
            w[k] += need / len(sb)
    return _normalize(w)


def _apply_asset_caps(w, uni):
    for k in w:
        lo, hi = uni[k]["cap"]
        w[k] = max(lo, min(hi, w[k]))
    return _normalize(w)


def compute_weights(scores, momentum, cfg, regime=None):
    """
    scores: {'risk_off','rates_up','inflation','fx_stress'} (z-score)
    momentum: {ticker: (1mo+3mo)/2 수익률}
    반환: {'regime', 'weights'(dict, 합 1), 'scores'(clip된)}
    """
    uni = cfg["universe"]
    if regime is None:
        regime = classify_regime(scores, cfg)
    sw = cfg["score_weights"]
    ro = max(_clip(scores.get("risk_off", 0)), 0.0)
    ru = max(_clip(scores.get("rates_up", 0)), 0.0)
    inf = max(_clip(scores.get("inflation", 0)), 0.0)
    fx = max(_clip(scores.get("fx_stress", 0)), 0.0)
    mcfg = cfg["momentum"]

    w = {}
    for k, meta in uni.items():
        grp = meta["group"]
        m = 1.0
        if grp in ("growth", "income"):
            m *= (1 - sw["risk_off_equity"] * ro)
        if grp == "longbond":
            m *= (1 - sw["rates_longbond"] * ru)
        if grp == "gold":
            m *= (1 + sw["inflation_gold"] * inf)
        if grp in ("shortbond", "cash"):
            m *= (1 + sw["riskoff_defensive"] * ro)
        # 약한 모멘텀 틸트 (현금 제외)
        if k != "CASH_KRW":
            mom = momentum.get(k, 0.0)
            if mom > mcfg["up"]:
                m *= mcfg["strong"]
            elif mom < mcfg["down"]:
                m *= mcfg["weak"]
        w[k] = max(0.0, meta["base"] * m)

    w = _apply_regime_caps(w, uni, cfg, regime)
    w = _apply_asset_caps(w, uni)
    return {
        "regime": regime,
        "weights": w,
        "scores": {"risk_off": ro, "rates_up": ru, "inflation": inf, "fx_stress": fx},
    }


# ───────────────────────── 데이터 fetch (네트워크) ─────────────────────────
def fetch_market_data(cfg):
    """yfinance로 매크로 점수 + 종목 모멘텀 계산. (Colab/GitHub Actions에서 실행)"""
    import yfinance as yf
    import pandas as pd

    mp = cfg["macro_proxies"]
    ymap = cfg["yf_map"]
    tickers = list(set(list(mp.values()) + list(ymap.values())))
    raw = yf.download(tickers, period="2y", progress=False, auto_adjust=False, group_by="ticker")

    def close(tkr):
        try:
            s = raw[tkr]["Close"].dropna()
            return s
        except Exception:
            return pd.Series(dtype=float)

    def ret_series(s, win):
        return (s / s.shift(win) - 1).dropna().tolist()

    def chg_series(s, win):
        return (s - s.shift(win)).dropna().tolist()

    def vol_series(s, win):
        return s.pct_change().rolling(win).std().dropna().tolist()

    # 점수 계산
    us10y = close(mp["us10y"]); wti = close(mp["wti"]); gold = close(mp["gold"])
    usdkrw = close(mp["usdkrw"]); ndq = close(mp["nasdaq"]); ks = close(mp["kospi"])
    krbond = close(ymap.get("148070", ""))  # 국고채10년 ETF (가격↓ = 금리↑)

    rates_up = _zscore_latest(chg_series(us10y, 63))
    if len(krbond) > 60:
        rates_up += -_zscore_latest(ret_series(krbond, 63))  # KR 금리 프록시
    inflation = _zscore_latest(ret_series(wti, 63)) + _zscore_latest(ret_series(gold, 63))
    risk_off = _zscore_latest(vol_series(ndq, 20)) + _zscore_latest(vol_series(ks, 20))
    fx_stress = _zscore_latest(ret_series(usdkrw, 63))

    scores = {"risk_off": risk_off, "rates_up": rates_up, "inflation": inflation, "fx_stress": fx_stress}

    # 모멘텀 ((1mo+3mo)/2)
    momentum = {}
    for code, yt in ymap.items():
        s = close(yt)
        if len(s) > 64:
            r1 = float(s.iloc[-1] / s.iloc[-21] - 1)
            r3 = float(s.iloc[-1] / s.iloc[-63] - 1)
            momentum[code] = (r1 + r3) / 2
        else:
            momentum[code] = 0.0
    return scores, momentum
