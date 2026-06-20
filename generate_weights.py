"""
최종 모델 비중 생성기 — 매크로 헷지(모멘텀 + 매크로 틸트, 국내 ETF)
비교 결과 Sharpe 2.42 / MDD -4.5%로 채택된 안정형 모델.

이 스크립트는 최신 목표비중을 계산해 Google Sheet의 'TargetWeights' 탭에 기록합니다.
그러면 앱스 스크립트가 그 비중을 읽어 매매합니다.

실행 환경 2가지를 자동 감지:
  - GitHub Actions: 환경변수 GCP_SA_KEY(서비스계정 JSON), SHEET_ID 사용
  - Colab: google.colab 대화형 인증 사용
"""
import os, json
import numpy as np
import pandas as pd
import yfinance as yf

# ── 최종 모델 파라미터 (GAS computeTargetWeights_와 동일) ──
ASSETS = ["133690.KS", "132030.KS", "305080.KS", "153130.KS", "069500.KS"]
NAME = {"133690.KS": "나스닥100", "132030.KS": "금", "305080.KS": "미국채10년",
        "153130.KS": "단기채", "069500.KS": "KOSPI200"}
BASE = {"133690.KS": .25, "132030.KS": .22, "305080.KS": .15, "153130.KS": .15, "069500.KS": .10}
# 매크로 틸트(role 기반): 금↑(고유가), 미국채10년↓(고유가·미금리), 나스닥↓(미금리), KOSPI↓(BOJ·한은)
TILT = {"133690.KS": 0.9, "132030.KS": 1.3, "305080.KS": 0.7 * 0.8, "153130.KS": 1.0, "069500.KS": 0.85 * 0.85}
RISK_MULT = 0.835   # 레짐 점수 ~-0.325 → 위험자산 배율
CAP = 0.90          # 위험자산 상한
SHEET_ID = os.environ.get("SHEET_ID", "1IeUupk0pAwDw6tBK-A7-12eNakwfLeo9F1cVdUqYIyk")


def momentum_tilt(s):
    """1·3개월 수익률 평균으로 모멘텀 틸트 (GAS momentumTilt_와 동일)"""
    if len(s) < 65:
        return 1.0
    m1 = s.iloc[-1] / s.iloc[-21] - 1
    m3 = s.iloc[-1] / s.iloc[-61] - 1
    m = (m1 + m3) / 2
    if m > 0.05:
        return 1.15
    if m < -0.05:
        return 0.85
    return 1.0


def compute_weights():
    px = yf.download(ASSETS, period="9mo", progress=False, auto_adjust=False)
    px = (px["Adj Close"] if "Adj Close" in px.columns else px["Close"]).ffill().dropna()
    w = {t: BASE[t] * TILT[t] * momentum_tilt(px[t]) for t in ASSETS}
    rs = sum(w.values())
    scaled = min(rs * RISK_MULT, CAP)
    norm = scaled / rs if rs > 0 else 0
    w = {t: round(v * norm, 4) for t, v in w.items()}
    used = sum(w.values())
    w["CASH_KRW"] = round(max(1 - used, 0), 4)   # 현금 잔여 = 방어분
    return w


def gspread_client():
    import gspread
    sa = os.environ.get("GCP_SA_KEY")
    if sa:  # GitHub Actions: 서비스계정
        from google.oauth2.service_account import Credentials
        info = json.loads(sa)
        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        return gspread.authorize(creds)
    # Colab: 대화형 인증
    from google.colab import auth
    auth.authenticate_user()
    from google.auth import default
    creds, _ = default()
    return gspread.authorize(creds)


def write_sheet(w):
    gc = gspread_client()
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet("TargetWeights")
    except Exception:
        ws = sh.add_worksheet("TargetWeights", rows=20, cols=6)
    ws.clear()
    rows = [["ticker", "weight"]] + [[t, w[t]] for t in w]   # RAW: 069500 등 0시작 코드 보존
    ws.update("A1", rows, value_input_option="RAW")
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    # E1 = 갱신시각 (GAS 신선도 가드가 이 셀을 읽음)
    ws.update("D1", [["updated", ts, "final:macro-hedge-momentum"]], value_input_option="RAW")
    return ts


if __name__ == "__main__":
    w = compute_weights()
    print("최종 목표비중:")
    for t, v in w.items():
        print(f"  {NAME.get(t, t):<10} {t:<10} {v*100:5.1f}%")
    ts = write_sheet(w)
    print("TargetWeights 시트 기록 완료 @", ts)
