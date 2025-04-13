# Fat Fingers 🧠💥

Welcome to the GitHub repository of **Fat Fingers**, our team for the **IMC Prosperity 3 (2025)** algorithmic trading competition.

We are a group of **5 French Financial Engineering and Applied Mathematics students from CY Tech**, combining quantitative reasoning, statistical modeling, and coding to build high-frequency trading strategies in a simulated multi-asset market.

---

## 🏝️ the competition

**IMC Prosperity 3 – 2025** is an algorithmic trading competition that lasted over 15 days, with over **9000 teams participating globally**.

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
| **Justin Léon**              | [🔗](https://www.linkedin.com/in/justin-l%C3%A9on/) |
| **Dorian Beurthe**           | [🔗](https://www.linkedin.com/in/dorian-beurthe-4a9a772b3/) |
| **Gabriel Tran-Phat**        | [🔗](https://www.linkedin.com/in/gabriel-tran-phat-751477317/) |
| **Julien Ruiz**              | [🔗](https://www.linkedin.com/in/julien-ruiz75/) |

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

- ✅ Currently in progress
- 🧪 Local backtest peak: **112,004 seashells**
- 📈 IMC evaluation score: **13,500 seashells**

**Strategy:**
- EMA + recent mean **hybrid fair value**
- **Volatility-aware spreads**
- **Confidence-scaled order sizes**
- **Basket arbitrage** on PICNIC_BASKET1 and PICNIC_BASKET2
- **Momentum filtering** to avoid fading strong directional moves
- Strict **position limit management**

We are now experimenting with directional bias models, VWAP layers, and aggressive execution logic for high-volatility assets like CROISSANTS.

---

## 🛠️ tools & tech

We used a full suite of testing and visualization tools to build, debug, and optimize our strategies:

- [✅ Prosperity 3 Wiki](https://imc-prosperity.notion.site/Prosperity-3-Wiki-19ee8453a09380529731c4e6fb697ea4)
- [✅ JMerle’s Backtester](https://github.com/jmerle/imc-prosperity-3-backtester)
- [✅ JMerle’s Visualizer](https://github.com/jmerle/imc-prosperity-3-visualizer)
- [✅ JMerle’s Submitter](https://github.com/jmerle/imc-prosperity-3-submitter)

💙 Huge thanks to [**@jmerle**](https://github.com/jmerle) for making his tools public and battle-tested. These made the IMC competition **actually fun to engineer for**.

---

## 📂 repository structure

