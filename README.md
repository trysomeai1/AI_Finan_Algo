# 1. 필수 라이브러리 설치
!pip install yfinance scikit-learn

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------
# 2. 설정: 투자 대상 및 기간
# -----------------------------------------------------------
SIM_START_DATE = '2023-01-01'  # 시뮬레이션 시작일
TRAIN_START_DATE = '2015-01-01' # AI 학습용 과거 데이터 시작일

# (A) AI 포트폴리오 구성 자산
portfolio_weights = {
    'QQQ': 0.35,       # 미국 기술주 (나스닥 100)
    'TLT': 0.25,       # 미국 장기채
    'GLD': 0.20,       # 금
    '069500.KS': 0.10, # 한국 주식 (코스피 200)
    'SHY': 0.10        # 현금성 자산
}

# (B) 비교할 벤치마크 (단독 투자 시)
# 사용자의 요청: 나스닥, 다우, 코스닥, 코스피
benchmarks = {
    'QQQ': 'Nasdaq 100 (US Tech)',
    'DIA': 'Dow Jones (US Blue Chip)', # 다우존스
    '229200.KS': 'KOSDAQ 150 (Korea Small Cap)', # 코스닥 150
    '069500.KS': 'KOSPI 200 (Korea Large Cap)'   # 코스피 200
}

# 전체 티커 리스트 (중복 제거)
all_tickers = list(set(list(portfolio_weights.keys()) + list(benchmarks.keys())))

# -----------------------------------------------------------
# 3. 데이터 다운로드 및 전처리
# -----------------------------------------------------------
def prepare_data():
    print("데이터 다운로드 및 전처리 중...")
    # 환율(KRW=X) 포함하여 다운로드
    data = yf.download(all_tickers + ['KRW=X'], start=TRAIN_START_DATE, progress=False, auto_adjust=False)

    # Adj Close 우선 사용
    if 'Adj Close' in data.columns:
        df = data['Adj Close']
    else:
        df = data['Close']

    # 환율 분리
    if 'KRW=X' in df.columns:
        rate = df['KRW=X'].fillna(method='ffill')
        df = df.drop(columns=['KRW=X'])
    else:
        rate = pd.Series(1300, index=df.index) # 없을 경우 기본값

    return df.dropna(), rate.dropna()

raw_df, rate_df = prepare_data()

# -----------------------------------------------------------
# 4. AI 학습 데이터 생성 (Feature Engineering)
# -----------------------------------------------------------
def make_features(price_df, lookahead_days):
    features = {}
    for t in price_df.columns:
        p = price_df[t]
        f = pd.DataFrame(index=p.index)
        f['Price'] = p

        # 기술적 지표 (입력값)
        f['Ret_1M'] = p.pct_change(20)
        f['Ret_3M'] = p.pct_change(60)
        f['Vol_1M'] = p.pct_change().rolling(20).std()

        # RSI
        delta = p.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        f['RSI'] = 100 - (100 / (1 + rs))

        # 목표값 (Target): n일 뒤 가격 상승 여부 (1/0)
        f['Target'] = (p.shift(-lookahead_days) > p).astype(int)

        features[t] = f.dropna()
    return features

# -----------------------------------------------------------
# 5. AI 백테스팅 엔진 (월간/2주간)
# -----------------------------------------------------------
def run_ai_backtest(freq, monthly_invest_amount, lookahead_days, label):
    print(f">> AI 시뮬레이션: {label} (주기: {freq})")

    feature_sets = make_features(raw_df, lookahead_days)

    # 리밸런싱 날짜 (주기별)
    rebalance_dates = pd.date_range(start=SIM_START_DATE, end=raw_df.index[-1], freq=freq)

    # 투자금 계산 (월간=100만, 2주간=50만 정도로 조정)
    # freq가 'MS'(월초)면 그대로, '2W'(2주)면 절반
    per_period_invest = monthly_invest_amount if freq == 'MS' else monthly_invest_amount / 2

    portfolio_value = []
    invested_capital = []

    current_shares = {t: 0 for t in portfolio_weights.keys()}
    cash_balance = 0
    total_invested = 0

    for date in rebalance_dates:
        # 데이터 시점 맞추기
        valid_idx = raw_df.index[raw_df.index <= date]
        if len(valid_idx) == 0: continue
        curr_date = valid_idx[-1]

        # 1. 투자금 투입
        cash_balance += per_period_invest
        total_invested += per_period_invest

        # 2. AI 예측 및 비중 결정
        target_weights = {}

        for t in portfolio_weights.keys():
            if t == 'SHY':
                target_weights[t] = portfolio_weights[t]
                continue

            data_t = feature_sets[t]
            train_data = data_t[data_t.index < curr_date] # 과거 데이터만 사용

            if len(train_data) < 100:
                target_weights[t] = portfolio_weights[t]
                continue

            # 모델 학습
            model = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=42)
            X = train_data[['Ret_1M', 'Ret_3M', 'Vol_1M', 'RSI']]
            y = train_data['Target']

            model.fit(X, y)

            # 예측
            last_row = X.iloc[[-1]]
            prob_up = model.predict_proba(last_row)[0][1]

            # 비중 조절 (상승확률 높으면 1.5배, 낮으면 0.5배)
            w = portfolio_weights[t]
            if prob_up > 0.6:   target_weights[t] = w * 1.5
            elif prob_up < 0.4: target_weights[t] = w * 0.5
            else:               target_weights[t] = w

        # 비중 정규화 (나머지는 현금 SHY)
        risky_sum = sum([w for t, w in target_weights.items() if t != 'SHY'])
        target_weights['SHY'] = max(1.0 - risky_sum, 0)

        # 3. 매매 실행 (리밸런싱)
        # 현재 자산 가치 계산
        curr_val = cash_balance
        prices = {}
        rates = {}

        for t, shares in current_shares.items():
            p = raw_df.loc[curr_date, t]
            r = rate_df.loc[curr_date] if '.KS' not in t else 1
            prices[t] = p
            rates[t] = r
            curr_val += shares * p * r

        # 목표 비중대로 매매
        for t, w in target_weights.items():
            target_amt = curr_val * w
            current_amt = current_shares[t] * prices[t] * rates[t]
            diff = target_amt - current_amt

            if diff > 0: # 매수
                amt = min(diff, cash_balance) if cash_balance > 0 else 0
                if amt > 0:
                    current_shares[t] += amt / (prices[t] * rates[t])
                    cash_balance -= amt
            else: # 매도
                amt = abs(diff)
                current_shares[t] -= amt / (prices[t] * rates[t])
                cash_balance += amt

        portfolio_value.append(curr_val)
        invested_capital.append(total_invested)

    return pd.DataFrame({
        f'{label}_Value': portfolio_value,
        f'{label}_Invested': invested_capital
    }, index=rebalance_dates)

# -----------------------------------------------------------
# 6. 벤치마크 적립식 투자 시뮬레이션 (비교군)
# -----------------------------------------------------------
def run_benchmark_dca(ticker, monthly_invest_amount, label):
    # 매월 초 투자
    dates = pd.date_range(start=SIM_START_DATE, end=raw_df.index[-1], freq='MS')

    values = []
    shares = 0
    total_invested = 0

    for date in dates:
        valid_idx = raw_df.index[raw_df.index <= date]
        if len(valid_idx) == 0: continue
        curr_date = valid_idx[-1]

        # 투자금 투입 및 매수
        total_invested += monthly_invest_amount

        p = raw_df.loc[curr_date, ticker]
        r = rate_df.loc[curr_date] if '.KS' not in ticker else 1

        # 환율 적용하여 매수 수량 계산
        shares_bought = monthly_invest_amount / (p * r)
        shares += shares_bought

        # 현재 평가액
        curr_val = shares * p * r
        values.append(curr_val)

    return pd.DataFrame({
        f'{label}': values
    }, index=dates)

# -----------------------------------------------------------
# 7. 실행 및 결과 통합
# -----------------------------------------------------------
# (1) AI 포트폴리오 실행
res_ai_monthly = run_ai_backtest('MS', 1000000, 20, 'AI Portfolio (Monthly)')
res_ai_biweekly = run_ai_backtest('2W-FRI', 1000000, 10, 'AI Portfolio (Bi-weekly)')

# (2) 벤치마크 실행 (나스닥, 다우, 코스닥, 코스피)
res_benchmarks = []
for t, name in benchmarks.items():
    res = run_benchmark_dca(t, 1000000, name)
    res_benchmarks.append(res)

# 데이터 병합 (날짜 인덱스 기준 정렬 및 채우기)
combined = pd.concat([res_ai_monthly, res_ai_biweekly] + res_benchmarks, axis=1).sort_index()
combined = combined.fillna(method='ffill').dropna()

# -----------------------------------------------------------
# 8. 최종 시각화 및 성과표
# -----------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(14, 8))

# AI 포트폴리오 (굵은 선)
plt.plot(combined.index, combined['AI Portfolio (Monthly)_Value'], label='AI Portfolio (Monthly)', linewidth=3, color='blue')
plt.plot(combined.index, combined['AI Portfolio (Bi-weekly)_Value'], label='AI Portfolio (Bi-weekly)', linewidth=3, color='red', linestyle='--')

# 벤치마크 (얇은 선)
colors = ['orange', 'green', 'purple', 'brown']
for i, (t, name) in enumerate(benchmarks.items()):
    plt.plot(combined.index, combined[name], label=f'Buy & Hold: {name}', alpha=0.6, linewidth=1.5, color=colors[i])

# 투자 원금
plt.plot(combined.index, combined['AI Portfolio (Monthly)_Invested'], label='Invested Principal', color='black', linestyle=':', alpha=0.5)

plt.title('AI Portfolio vs Market Benchmarks (DCA Simulation)', fontsize=16)
plt.ylabel('Portfolio Value (KRW)')
plt.legend()
plt.show()

# 최종 수익률 출력
print(f"\n[ 최종 성과 비교 ({SIM_START_DATE} ~ 현재) ]")
print("-" * 60)
final_invested = combined['AI Portfolio (Monthly)_Invested'].iloc[-1]

# 성과 정렬을 위한 리스트 생성
performance_list = []

# AI 성과 추가
val_m = combined['AI Portfolio (Monthly)_Value'].iloc[-1]
performance_list.append(('AI Portfolio (Monthly)', val_m))

val_w = combined['AI Portfolio (Bi-weekly)_Value'].iloc[-1]
performance_list.append(('AI Portfolio (Bi-weekly)', val_w))

# 벤치마크 성과 추가
for t, name in benchmarks.items():
    val = combined[name].iloc[-1]
    performance_list.append((name, val))

# 수익률 높은 순서대로 정렬
performance_list.sort(key=lambda x: x[1], reverse=True)

print(f"총 투자 원금: {final_invested:,.0f} 원\n")
rank = 1
for name, val in performance_list:
    profit = val - final_invested
    roi = (profit / final_invested) * 100
    print(f"{rank}. {name:<30} | 평가액: {val:,.0f}원 ({roi:+.2f}%)")
    rank += 1
print("-" * 60)
