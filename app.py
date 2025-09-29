import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'MS Gothic'
rcParams['axes.unicode_minus'] = False

N = 5000
inflation = 0.02

def generate_prices(year, mu, sigma, n_sim=N, steps_per_year=252):
    t = steps_per_year
    delta = 1/t
    days = year * t

    prices = np.zeros((n_sim, days + 1), dtype=np.float32)
    prices[:, 0] = 100

    for n in range(1, days + 1):
        prices[:, n] = prices[:, n-1] * ((1 + mu)**delta) * np.exp(sigma * np.sqrt(delta) * np.random.randn(n_sim))
    
    return prices[:, ::t]

def wilson_score_interval(n_success, n, z=1.96):
    phat = n_success / n
    SE = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4*n**2))
    denominator = 1 + z**2 / n
    lower, upper = (phat + z**2 / (2*n) - SE) / denominator, (phat + z**2 / (2*n) + SE) / denominator
    return lower, upper

def fixed_amount_method(prices, first_asset, fixed_amount, inflation=inflation):

    n_sim = prices.shape[0]
    year = prices.shape[1] - 1

    asset = np.zeros((n_sim, year + 1), dtype=np.float32)
    asset[:, 0] = first_asset
    unit = first_asset / prices[:, 0] #(n_sim)

    fixed_amount_annual = fixed_amount * ((1 + inflation) ** np.arange(0, year + 1))
    total_withdraw = np.zeros(n_sim, dtype=np.float32)

    for t in range(1, year + 1):

        asset[:, t] = np.maximum(0.0, unit * prices[:, t])
        withdraw = np.minimum(asset[:, t], fixed_amount_annual[t])
        unit -= withdraw / prices[:, t]

    return asset, total_withdraw

def fixed_rate_method(prices, first_asset, withdraw_rate, live_expense, inflation=inflation):

    """
    live_expense: (2, )
    """

    n_sim = prices.shape[0]
    year = prices.shape[1] - 1

    asset = np.zeros((n_sim, year + 1), dtype=np.float32)
    asset[:, 0] = first_asset
    unit = first_asset / prices[:, 0] #(n_sim)

    live_expense_annual = live_expense[None, :] * ((1 + inflation) ** np.arange(0, year + 1)[:, None]) #(year + 1, 2)
    total_withdraw = np.zeros(n_sim, dtype=np.float32)

    for t in range(1, year + 1):

        asset[:, t] = np.maximum(0.0, unit * prices[:, t])
        withdraw = np.maximum(live_expense_annual[t, 0], np.minimum(live_expense_annual[t, 1], asset[:, t] * withdraw_rate))
        final_withdraw = np.minimum(asset[:, t], withdraw)
        unit -= final_withdraw / prices[:, t]
        
        total_withdraw += final_withdraw / (1 + inflation)**t

    return asset, total_withdraw

def calc_statistics(asset, total_withdraw, inflation=inflation):
    
    n_sim = asset.shape[0]
    year = asset.shape[1] - 1
    
    failure_flag = asset[:, -1] == 0
    n_failure = failure_flag.sum()
    
    failure_rate_transition = np.zeros(year + 1, dtype=np.float32)
    for t in range(1, year + 1):
        failure_flag_transition = asset[:, t] == 0
        n_failure_transition = failure_flag_transition.sum()
        failure_rate_transition[t] = n_failure_transition / n_sim
    
    lower, upper = wilson_score_interval(n_failure, n_sim)
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    percentile = [25, 50, 75]
    p_asset = np.array([np.percentile(asset[:, t], percentile) / ((1 + inflation)**t) for t in range(year + 1)]) #(year + 1)

    withdraw_mid = np.percentile(total_withdraw, 50)

    return [failure_rate_transition, lower, upper, p_asset, withdraw_mid]

funds = {
    "全世界株式（VT）": {"mu": 6.0, "sigma": 15.0},
    "S&P500（VOO）": {"mu": 10.0, "sigma": 20.0}
}

st.title("取り崩しシミュレーション")
st.text("このアプリは初期資産額等の情報を入力すれば年毎の失敗率や資産額の推移を算出できます。")
st.text("なお、あくまで数理的なシミュレーションです。投資判断は自己責任でお願いします。")
st.markdown("---")

fst_asset = st.number_input("初期資産額[万円]を入力してください", value=0)

st.text("")
year = st.slider("取り崩し年数を選択してください", min_value=0, max_value=120, value=0)

st.text("")
option1 = st.radio("FIRE戦略を選択してください", ("定額法", "定率法"), horizontal=True)

if option1=="定額法":
    st.text("")
    w = st.number_input("毎年の取り崩し額[万円]を入力してください", value=0)

else:
    st.text("")
    min_live_expense = st.number_input("毎年の取り崩す下限の金額[万円]を入力してください。", value=0, help="毎年の絶対に取り崩したいお金のことです。0だと失敗率は0%になります。")
    max_live_expense = st.number_input("毎年の取り崩す上限の金額[万円]を入力してください。下限以下の場合は上限なしとなります。", value=0, help="毎年の取り崩したいお金の上限のことです")
    if max_live_expense <= min_live_expense:
        max_live_expense = float("inf")

    st.text("")
    rate = st.slider("取り崩し率[%]を選択してください", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

st.text("")
mu = st.number_input("年率想定リターン[%]", value=st.session_state.get("mu", 0.0))
sigma = st.number_input("年率想定リスク[%]", value=st.session_state.get("sigma", 0.0))

if "prev_fund" not in st.session_state:
    st.session_state.prev_fund = "選択しない"
st.text("")
select_fund = st.selectbox("リスク・リターンを参考値から選ぶ", ["選択しない"] + list(funds.keys()))
if select_fund != "選択しない" and select_fund != st.session_state.prev_fund:
    st.session_state.mu = funds[select_fund]["mu"]
    st.session_state.sigma = funds[select_fund]["sigma"]
    st.session_state.prev_fund = select_fund
    st.rerun()

if st.button("シミュレーション開始"):

    if fst_asset == 0:
        st.warning("初期資産額が0円です!")
    elif year == 0:
        st.warning("取り崩し期間が0年です!")

    else:
        st.markdown("---")
        prices = generate_prices(year, mu/100.0, sigma/100.0)
        if option1=="定額法":
            asset, total_withdraw = fixed_amount_method(prices, fst_asset, w)
        else:
            live_expense = np.array([min_live_expense, max_live_expense])
            asset, total_withdraw = fixed_rate_method(prices, fst_asset, rate/100, live_expense)
        result = calc_statistics(asset, total_withdraw)

        failure_rate_transition = result[0]
        lower = result[1]
        upper = result[2]
        p_asset = result[3]
        withdraw_mid = result[4]

        failure_rate = failure_rate_transition[-1]
        fig, ax = plt.subplots()
        ax.plot(failure_rate_transition * np.full_like(failure_rate_transition, 100.0), label="失敗率", color="red")
        ax.set_title("失敗率推移")
        ax.set_xlabel("年数")
        ax.set_ylabel("確率[%]")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.text(f"失敗率:{failure_rate*100.0:.1f}%", help="あくまで推定値です。正確な値はわからないので下の信頼区間を見てください。")
        st.text(f"失敗率95%信頼区間:({lower*100.0:.1f}% ~ {upper*100.0:.1f}%)", help="だいたいこのくらいの範囲の失敗率と予想できます")

        fig, ax = plt.subplots()
        fst_asset_annual = np.full(year + 1, fst_asset) / ((1 + inflation)**np.arange(0, year + 1))
        ax.fill_between(np.arange(year + 1), fst_asset_annual, np.zeros(year + 1), label="初期資産", color="gray")
        ax.plot(p_asset[:, 2], label="成績が上位25%", color="red")
        ax.plot(p_asset[:, 1], label="想定成績", color="green")
        ax.plot(p_asset[:, 0], label="成績が下位25%", color="blue")

        ax.set_title("インフレ調整付き資産額推移")
        ax.set_xlabel("年数")
        ax.set_ylabel("資産額[万円]")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        if option1 == "定率法":
            st.text(f"総取り崩し額中央値:{int(withdraw_mid)}万円", help="インフレ（物価上昇）の影響を調整した値です")
            st.text(f"総取り崩し額中央値年平均：{int(withdraw_mid / year)}万円", help="毎年の取り崩すお金の平均値です")
