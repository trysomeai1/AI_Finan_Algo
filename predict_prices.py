"""
ML 단기 가격 예측 → 'Predictions' 시트 기록 (수동 매매 보조 정보)

※ 자동매매(모멘텀 매크로 헷지)는 이 예측을 사용하지 않습니다 — production 모델은 그대로 유지.
※ 단기 가격 예측은 본질적으로 신뢰도가 낮습니다. 5일 예측의 과거 평균오차(mae5d_pct)를
   함께 기록해 불확실성을 표시합니다. 점 추정치를 맹신하지 마세요.

종목별로 기록: 현재가, 전일 고가/저가, 내일 예측가, 5일 뒤 예측가, 5일예측 평균오차(%)
실행 환경 자동 감지: GitHub Actions(GCP_SA_KEY) / Colab(대화형 인증)
"""
import os, json
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor

ASSETS = ["360750.KS", "133690.KS", "458730.KS", "241180.KS", "069500.KS",
          "329200.KS", "132030.KS", "305080.KS", "148070.KS", "153130.KS"]
NAME = {"360750.KS": "TIGER 미국S&P500", "133690.KS": "TIGER 미국나스닥100",
        "458730.KS": "TIGER 미국배당다우존스", "241180.KS": "TIGER 일본니케이225",
        "069500.KS": "KODEX 200", "329200.KS": "TIGER 리츠부동산인프라",
        "132030.KS": "KODEX 골드선물(H)", "305080.KS": "TIGER 미국채10년선물",
        "148070.KS": "KIWOOM 국고채10년", "153130.KS": "KODEX 단기채권"}
SHEET_ID = os.environ.get("SHEET_ID", "1IeUupk0pAwDw6tBK-A7-12eNakwfLeo9F1cVdUqYIyk")
H5 = 5  # 5거래일 예측


def tech(s, pre=""):
    f = pd.DataFrame(index=s.index)
    f[pre + "r5"] = s.pct_change(5)
    f[pre + "r21"] = s.pct_change(21)
    f[pre + "r63"] = s.pct_change(63)
    d = s.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    rs = up.rolling(14).mean() / (dn.rolling(14).mean() + 1e-9)
    f[pre + "rsi"] = 100 - 100 / (1 + rs)
    e12 = s.ewm(span=12).mean(); e26 = s.ewm(span=26).mean(); macd = e12 - e26
    f[pre + "macdh"] = (macd - macd.ewm(span=9).mean()) / s
    f[pre + "ma20"] = s / s.rolling(20).mean() - 1
    f[pre + "ma60"] = s / s.rolling(60).mean() - 1
    f[pre + "vol"] = s.pct_change().rolling(20).std()
    return f


def dl(tk, field="Close"):
    r = yf.download(tk, period="4y", progress=False, auto_adjust=False)
    lvl0 = r.columns.get_level_values(0) if hasattr(r.columns, "get_level_values") else r.columns
    col = field if field in list(lvl0) else "Close"
    return r[col].ffill()


def predict():
    px = dl(ASSETS, "Close").ffill().dropna()
    hi = dl(ASSETS, "High"); lo = dl(ASSETS, "Low")
    oil = dl("CL=F", "Close"); cu = dl("HG=F", "Close")
    of = tech(oil, "oil_").reindex(px.index, method="ffill")
    cf = tech(cu, "cu_").reindex(px.index, method="ffill")
    feat = {t: tech(px[t]).join(of).join(cf) for t in ASSETS}
    FCOLS = list(feat[ASSETS[0]].columns)
    rows = []
    for t in ASSETS:
        df = feat[t]; s = px[t]
        y1 = s.shift(-1) / s - 1
        y5 = s.shift(-H5) / s - 1

        def fit(y):
            dd = pd.concat([df[FCOLS], y.rename("y")], axis=1).dropna()
            n = len(dd); k = int(n * 0.8); mae = float("nan")
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, random_state=42)
            if n - k > 5:
                m.fit(dd[FCOLS].iloc[:k], dd["y"].iloc[:k])
                pr = m.predict(dd[FCOLS].iloc[k:])
                mae = float(np.mean(np.abs(pr - dd["y"].iloc[k:].values)))
            m.fit(dd[FCOLS], dd["y"])
            xi = df[FCOLS].iloc[[-1]]
            p = float(m.predict(xi)[0]) if not xi.isna().any().any() else 0.0
            return p, mae

        p1, _ = fit(y1)
        p5, mae5 = fit(y5)
        cur = float(s.iloc[-1])
        rows.append([t, round(cur), round(float(hi[t].iloc[-1])), round(float(lo[t].iloc[-1])),
                     round(cur * (1 + p1)), round(cur * (1 + p5)), round((mae5 or 0) * 100, 1)])
    return rows


def gspread_client():
    import gspread
    sa = os.environ.get("GCP_SA_KEY")
    if sa:
        from google.oauth2.service_account import Credentials
        c = Credentials.from_service_account_info(
            json.loads(sa), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        return gspread.authorize(c)
    from google.colab import auth
    auth.authenticate_user()
    from google.auth import default
    c, _ = default()
    return gspread.authorize(c)


def write_sheet(rows):
    gc = gspread_client()
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet("Predictions")
    except Exception:
        ws = sh.add_worksheet("Predictions", rows=20, cols=10)
    ws.clear()
    header = [["ticker", "current", "prevHigh", "prevLow", "pred1d", "pred5d", "mae5d_pct"]]
    ws.update("A1", header + rows, value_input_option="RAW")
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    ws.update("I1", [["updated", ts]], value_input_option="RAW")
    return ts


if __name__ == "__main__":
    rows = predict()
    print("ML 단기 예측 (참고용):")
    for r in rows:
        print(f"  {NAME.get(r[0], r[0]):<10} 현재 {r[1]:>8,}  전일 고/저 {r[2]:,}/{r[3]:,}  "
              f"내일 {r[4]:,}  5일 {r[5]:,} (±{r[6]}%)")
    ts = write_sheet(rows)
    print("Predictions 시트 기록 완료 @", ts)
