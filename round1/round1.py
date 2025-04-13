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
    "DJEMBES": 50,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 60
}

# Proportions des composants dans les paniers
BASKET1_COMPONENTS = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
BASKET2_COMPONENTS = {"CROISSANTS": 4, "JAMS": 2, "DJEMBES": 4}

ALPHA = 0.2  # Coefficient pour moyenne exponentielle

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

        # Mise à jour EMA (moyenne exponentielle) des justes valeurs
        fair_values = {}
        for product in state.order_depths:
            trades = state.market_trades.get(product, [])
            ema_key = f"{product}_EMA"
            old_ema = trader_memory.get(ema_key, 100)
            for trade in trades:
                old_ema = ALPHA * trade.price + (1 - ALPHA) * old_ema
            fair_values[product] = old_ema
            trader_memory[ema_key] = old_ema

        # Market making actif sur tous les produits simples
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK", "CROISSANTS", "JAMS", "DJEMBES"]:
            orders = self.aggressive_market_maker(product, state.order_depths[product],
                                                  fair_values[product], state.position.get(product, 0),
                                                  POSITION_LIMITS[product])
            if orders:
                result[product] = orders

        # Arbitrage complet sur PICNIC_BASKET1 et 2
        result["PICNIC_BASKET1"] = self.basket_arbitrage("PICNIC_BASKET1", BASKET1_COMPONENTS,
                                                          state, fair_values)
        result["PICNIC_BASKET2"] = self.basket_arbitrage("PICNIC_BASKET2", BASKET2_COMPONENTS,
                                                          state, fair_values)

        return result, conversions, jsonpickle.encode(trader_memory)

    def aggressive_market_maker(self, product, order_depth, fair_value, position, limit):
        spread = max(1, int(fair_value * 0.015))  # 1.5% spread
        orders = []

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

    def basket_arbitrage(self, basket_name, components, state, fair_values):
        orders = []
        position = state.position.get(basket_name, 0)
        order_depth = state.order_depths[basket_name]
        limit = POSITION_LIMITS[basket_name]

        component_value = sum(fair_values[prod] * qty for prod, qty in components.items())

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > component_value + 3:
                qty = min(bid_volume, limit + position)
                if qty > 0:
                    orders.append(Order(basket_name, bid_price, -qty))
                    position -= qty

        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price < component_value - 3:
                qty = min(-ask_volume, limit - position)
                if qty > 0:
                    orders.append(Order(basket_name, ask_price, qty))
                    position += qty

        return orders