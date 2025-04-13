from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import jsonpickle
from statistics import mean

# Configuration
POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 50,
    "JAMS": 50,
    "DJEMBES": 50
}

ALPHA = 0.2  # Coefficient pour moyenne exponentielle
MIN_SPREAD = {
    "RAINFOREST_RESIN": 1,
    "KELP": 2,
    "SQUID_INK": 5,
    "CROISSANTS": 1,
    "JAMS": 1,
    "DJEMBES": 1
}

class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_memory = {}

        if state.traderData:
            try:
                trader_memory = jsonpickle.decode(state.traderData)
            except:
                trader_memory = {}

        # Mise à jour EMA des justes valeurs
        fair_values = {}
        for product in state.order_depths:
            trades = state.market_trades.get(product, [])
            ema_key = f"{product}_EMA"
            old_ema = trader_memory.get(ema_key, 100)
            for trade in trades:
                old_ema = ALPHA * trade.price + (1 - ALPHA) * old_ema
            fair_values[product] = old_ema
            trader_memory[ema_key] = old_ema

        # Market making sur produits simples uniquement
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK", "CROISSANTS", "JAMS", "DJEMBES"]:
            orders = self.adaptive_market_maker(product, state.order_depths[product],
                                                fair_values[product], state.position.get(product, 0),
                                                POSITION_LIMITS[product], state.timestamp)
            if orders:
                result[product] = orders

        return result, conversions, jsonpickle.encode(trader_memory)

    def adaptive_market_maker(self, product, order_depth, fair_value, position, limit, timestamp):
        spread = max(MIN_SPREAD[product], int(fair_value * 0.015))
        orders = []

        # Ajustement d'ouverture (plus prudent au début)
        if timestamp < 20000:
            spread *= 1.5

        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price < fair_value - spread:
                qty = min(-ask_volume, limit - position)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    position += qty

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > fair_value + spread:
                qty = min(bid_volume, limit + position)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    position -= qty

        return orders

