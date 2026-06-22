"""
run_generate_weights.py — v3 비중 생성 엔트리포인트.

사용:
  # 로컬/시트 없이 CSV로 확인
  python src/run_generate_weights.py --config config/universe_kr_etf.json --out target_weights.csv --no-sheet
  # 시트에 기록 (env: GCP_SA_KEY=서비스계정 JSON, SHEET_ID=시트ID)
  python src/run_generate_weights.py --config config/universe_kr_etf.json

GitHub Actions에서는 GCP_SA_KEY / SHEET_ID 시크릿으로 시트에 직접 기록한다.
TargetWeights 시트에 [code, weight] 와 E1=timestamp, F1=regime, G1=scores 를 함께 남긴다.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import portfolio_engine as pe


def write_csv(path, result):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["code", "weight"])
        for k, v in result["weights"].items():
            wr.writerow([k, round(v, 6)])
    print(f"CSV 저장: {path}")


def write_sheet(result):
    import gspread
    from google.oauth2.service_account import Credentials

    sa = os.environ.get("GCP_SA_KEY")
    sheet_id = os.environ.get("SHEET_ID")
    if not sa or not sheet_id:
        raise SystemExit("GCP_SA_KEY / SHEET_ID 환경변수가 필요합니다 (또는 --no-sheet 사용).")
    info = json.loads(sa)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    gc = gspread.authorize(Credentials.from_service_account_info(info, scopes=scopes))
    ss = gc.open_by_key(sheet_id)
    try:
        ws = ss.worksheet("TargetWeights")
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet("TargetWeights", rows=50, cols=8)
    ws.clear()
    rows = [["code", "weight"]] + [[k, round(v, 6)] for k, v in result["weights"].items()]
    # RAW: 069500 등 앞자리 0 보존
    ws.update("A1", rows, value_input_option="RAW")
    kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
    ws.update("E1", [[kst]], value_input_option="RAW")
    ws.update("F1", [[result["regime"]]], value_input_option="RAW")
    ws.update("G1", [[json.dumps(result["scores"])]], value_input_option="RAW")
    print(f"시트 기록 완료 · regime={result['regime']} · {kst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config", "universe_kr_etf.json"))
    ap.add_argument("--out", default="target_weights.csv")
    ap.add_argument("--no-sheet", action="store_true", help="시트 대신 CSV만")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    scores, momentum = pe.fetch_market_data(cfg)
    result = pe.compute_weights(scores, momentum, cfg)

    print("regime :", result["regime"])
    print("scores :", {k: round(v, 2) for k, v in result["scores"].items()})
    print("weights:")
    for k, v in result["weights"].items():
        print(f"  {k:9s} {cfg['universe'][k]['name']:18s} {v*100:5.1f}%")

    if args.no_sheet:
        write_csv(args.out, result)
    else:
        write_sheet(result)


if __name__ == "__main__":
    main()
