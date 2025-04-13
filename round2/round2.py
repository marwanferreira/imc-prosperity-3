from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import jsonpickle
from statistics import mean, pstdev

# --- PARAMETERS ---
POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "CROISSANTS": 50, "JAMS": 50, "DJEMBES": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 60
}

BASKET1_COMPONENTS = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
BASKET2_COMPONENTS = {"CROISSANTS": 4, "JAMS": 2}

VOL_WINDOW = 20
EPSILON = 1e-6

# Regime thresholds
VOLATILITY_THRESHOLD = 5

class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        memory = {}

        if state.traderData:
            try:
                memory = jsonpickle.decode(state.traderData)
            except:
                memory = {}

        fair_values, volatilities, regimes = {}, {}, {}

        # === Fair Value & Volatility ===
        for product in state.order_depths:
            trades = state.market_trades.get(product, [])
            ema_key, hist_key = f"{product}_EMA", f"{product}_HIST"
            ema = memory.get(ema_key, 100)
            hist = memory.get(hist_key, [])

            for trade in trades:
                ema = 0.2 * trade.price + 0.8 * ema
                hist.append(trade.price)

            hist = hist[-VOL_WINDOW:]
            memory[ema_key], memory[hist_key] = ema, hist

            recent_mean = mean(hist) if hist else ema
            fair_value = 0.7 * ema + 0.3 * recent_mean
            fair_values[product] = fair_value

            volatility = pstdev(hist) if len(hist) > 1 else 1
            volatility = max(volatility, EPSILON)
            volatilities[product] = volatility

            # Regime detection
            regimes[product] = "volatile" if volatility > VOLATILITY_THRESHOLD else "stable"

        # === Market Making ===
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK", "CROISSANTS", "JAMS", "DJEMBES"]:
            orders = self.market_make(
                product,
                state.order_depths[product],
                fair_values[product],
                volatilities[product],
                regimes[product],
                state.position.get(product, 0),
                POSITION_LIMITS[product]
            )
            if orders:
                result[product] = orders

        # === Basket Arbitrage ===
        for basket, components in [("PICNIC_BASKET1", BASKET1_COMPONENTS), ("PICNIC_BASKET2", BASKET2_COMPONENTS)]:
            orders = self.basket_arbitrage(
                basket,
                components,
                state,
                fair_values,
                volatilities
            )
            if orders:
                result[basket] = orders

        return result, conversions, jsonpickle.encode(memory)

    def market_make(self, product: str, order_depth: OrderDepth, fair_value: float,
                    volatility: float, regime: str, position: int, limit: int) -> List[Order]:
        orders = []

        # Adaptive strategy based on regime
        if regime == "stable":
            spread_pct = 0.01
            confidence_scaling = 2.0
        else:
            spread_pct = 0.02  # give more buffer
            confidence_scaling = 1.0  # be less aggressive

        base_spread = max(1, int(volatility + fair_value * spread_pct))

        for ask, volume in sorted(order_depth.sell_orders.items()):
            if ask < fair_value - base_spread:
                confidence = abs(fair_value - ask) / max(volatility, EPSILON)
                qty = min(int(confidence * confidence_scaling), limit - position)
                if position > 0.8 * limit:
                    qty = min(1, qty)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    position += qty

        for bid, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > fair_value + base_spread:
                confidence = abs(fair_value - bid) / max(volatility, EPSILON)
                qty = min(int(confidence * confidence_scaling), limit + position)
                if position < -0.8 * limit:
                    qty = min(1, qty)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    position -= qty

        return orders

    def basket_arbitrage(self, basket_name: str, components: Dict[str, int],
                         state: TradingState, fair_values: Dict[str, float],
                         volatilities: Dict[str, float]) -> List[Order]:
        orders = []
        depth = state.order_depths[basket_name]
        position = state.position.get(basket_name, 0)
        limit = POSITION_LIMITS[basket_name]

        comp_value = sum(fair_values[p] * q for p, q in components.items())
        comp_vol = sum(volatilities[p] * q for p, q in components.items())
        comp_vol = max(comp_vol, EPSILON)
        threshold = max(3, int(comp_vol * 0.5))

        for bid_price, bid_volume in sorted(depth.buy_orders.items(), reverse=True):
            if bid_price > comp_value + threshold:
                confidence = abs(bid_price - comp_value) / comp_vol
                qty = min(int(confidence * 2), limit + position)
                if qty > 0:
                    orders.append(Order(basket_name, bid_price, -qty))
                    position -= qty

        for ask_price, ask_volume in sorted(depth.sell_orders.items()):
            if ask_price < comp_value - threshold:
                confidence = abs(comp_value - ask_price) / comp_vol
                qty = min(int(confidence * 2), limit - position)
                if qty > 0:
                    orders.append(Order(basket_name, ask_price, qty))
                    position += qty

        return orders
