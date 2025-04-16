
# Fat Fingers 👆

Welcome to the GitHub repository of **Fat Fingers**, our team for the **IMC Prosperity 3 (2025)** algorithmic trading competition.

We are a group of **5 French Financial Engineering and Applied Mathematics students from CY Tech**, combining quantitative reasoning, statistical modeling, and coding to build high-frequency trading strategies in a simulated multi-asset market.

---

## 🏝️ the competition

**IMC Prosperity 3 – 2025** is an algorithmic trading competition that lasted over 15 days, with over **10000 teams participating globally**.

In the challenge, we were tasked with **algorithmically trading various products**, such as amethysts, starfruit, orchids, coconuts, gift baskets, and more, with the goal of **maximizing seashells** – the currency of our island. The trading environment mimicked real markets, and we competed against **bots whose behavior could be inferred from historical data**.

At the end of each round, our algorithm was evaluated independently, and the resulting PnL ranked against all other teams.

In addition to the main algorithmic focus, there were also **manual trading mini-games**. These varied in nature and accounted for a small portion of total profit.

📚 For documentation on the trading environment and structure, refer to the official [Prosperity 3 Wiki](https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4).

---

## 👥 the team

We proudly represented **CY Tech, France** with a shared love for quant finance, data science, and algorithmic design.

| Name | LinkedIn |
|------|----------|
| **Marwan Ferreira da Silva** | [🔗](https://www.linkedin.com/in/marwan-ferreira-da-silva/) |
| **Julien Ruiz**              | [🔗](https://www.linkedin.com/in/julien-ruiz75/) |
| **Dorian Beurthe**           | [🔗](https://www.linkedin.com/in/dorian-beurthe-4a9a772b3/) |
| **Justin Léon**              | [🔗](https://www.linkedin.com/in/justin-l%C3%A9on/) |
| **Gabriel Tran-Phat**        | [🔗](https://www.linkedin.com/in/gabriel-tran-phat-751477317/) |

---

## 📊 our results

### Round 1: *Amethysts & Starfruit*

- 🐚 **49,762 seashells**
- 🌍 **1705th out of 10,000+ teams**
- 🕹️ **1467th in manual trading**
- 🇫🇷 **58th in France**

**Strategy:**  
We implemented a **fair-value market making model**, relying on constant mid-prices (for amethysts) and rolling average mid estimation (for starfruit). We adjusted our edge dynamically and handled inventory risk by flattening positions using 0 EV trades when available.

---

### Round 2: *CROISSANTS, BASKETS, and Multi-Product Arbitrage*

- 🐚 **78,348 seashells**
- 🌍 **937th place overall**
- 🕹️ **291st in manual trading**
- 🇫🇷 **38th in France**
- 📈 IMC evaluation score: **13,500 seashells**
- 🧪 Local backtest peak: **112,004 seashells**

**Strategy:**
- EMA + recent mean **hybrid fair value**
- **Volatility-aware spreads**
- **Confidence-scaled order sizes**
- **Basket arbitrage** on PICNIC_BASKET1 and PICNIC_BASKET2
- **Momentum filtering** to avoid fading strong directional moves
- Strict **position limit management**

We refined our strategy using live backtests and market simulations, focusing on capitalizing on temporary inefficiencies between components and their baskets. Our manual trading also improved significantly in this round.

---
## 🧠 Round 3: *Options, Delta Hedging & Regime-Switching Alpha*

- 🐚 **482,195 seashells**
- 🌍 **42nd place worldwide**
- 🇫🇷 **1st in France**
- 🕹️ **96th in manual trading**
- 📈 **IMC evaluation score:** Major improvement over Round 2  
- 📊 **Cumulative total profit:** **482,195 seashells**

---

## 🚀 Strategy Highlights

In Round 3, we extended our multi-product trading engine by integrating **options pricing**, **live delta hedging**, and **regime-adaptive behaviors**, resulting in a major boost in PnL and global ranking.

### Core Components:
- 🧮 **Black-Scholes options pricing** for theoretical valuation.
- ⚖️ **Delta hedging** using the underlying asset with frequent rebalancing.
- 📉📈 **Regime switching** based on real-time volatility and momentum:
  - Mean-reversion strategies in low-vol regimes
  - Trend-following logic in directional markets
- 📊 **Hybrid fair value estimation** combining EMA, short-term mean, and microstructure-aware adjustments.
- 🧠 **Confidence-weighted order sizing** based on signal strength & market regime.
- 🧺 **Basket arbitrage** on PICNIC_BASKET1 and PICNIC_BASKET2 using synthetic component value models.
- 🛡️ **Capital protection & drawdown control** through rolling position limits and live PnL risk checks.

---

## ⚙️ Engineering Refinements

- Modular, event-driven architecture for handling bursts with minimal latency.
- Rebuilt trade/message parsers for cleaner state management.
- Clear separation between valuation, signal generation, and execution logic.
- Stress-tested in multiple backtest environments: high-volatility, skewed pricing, and synthetic imbalance scenarios.

---

## 🛠️ tools & tech

We used a full suite of testing and visualization tools to build, debug, and optimize our strategies:

- [✅ Prosperity 3 Wiki](https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4)
- [✅ JMerle’s Backtester](https://github.com/jmerle/imc-prosperity-3-backtester)
- [✅ JMerle’s Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer)
- [✅ JMerle’s Submitter](https://github.com/jmerle/imc-prosperity-3-submitter)
- [✅ IMC Discord](https://discord.com/channels/1001852729725046804/1337359637128806490)

Huge thanks to [**@jmerle**](https://github.com/jmerle) for making his tools public and battle-tested. These made the IMC competition **actually fun to engineer for**.

---
