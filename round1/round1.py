from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import jsonpickle
from statistics import mean

POSITION_LIMIT = 50

FAIR_VALUES = {
    "RAINFOREST_RESIN": 100,
    "KELP": 100,  # sera corrigée dynamiquement plus tard
    "SQUID_INK": 100  # valeur de fallback
}

class Trader:
    def run(self, state: TradingState):
        result = {}
        trader_memory = {}

        # On décode la mémoire d'état précédente si elle existe
        if state.traderData:
            try:
                trader_memory = jsonpickle.decode(state.traderData)
            except:
                trader_memory = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)

            if product == "RAINFOREST_RESIN":
                orders = self.handle_resin(product, order_depth, position)

            elif product == "KELP":
                orders = self.handle_kelp(product, order_depth, position)

            elif product == "SQUID_INK":
                recent_prices = trader_memory.get("SQUID_INK_HISTORY", [])
                trades = state.market_trades.get(product, [])
                for trade in trades:
                    recent_prices.append(trade.price)
                recent_prices = recent_prices[-10:]
                trader_memory["SQUID_INK_HISTORY"] = recent_prices

                orders = self.handle_squid(product, order_depth, position, recent_prices)

            result[product] = orders

        traderData = jsonpickle.encode(trader_memory)
        conversions = 0
        return result, conversions, traderData

    def handle_resin(self, product, order_depth, position):
        fair_value = FAIR_VALUES[product]
        spread = 1
        orders = []

        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price < fair_value - spread:
                buy_qty = min(-ask_volume, POSITION_LIMIT - position)
                if buy_qty > 0:
                    orders.append(Order(product, ask_price, buy_qty))
                    position += buy_qty

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > fair_value + spread:
                sell_qty = min(bid_volume, POSITION_LIMIT + position)
                if sell_qty > 0:
                    orders.append(Order(product, bid_price, -sell_qty))
                    position -= sell_qty

        return orders

    def handle_kelp(self, product, order_depth, position):
        fair_value = FAIR_VALUES[product]
        spread = 2  # Kelp est plus volatil
        orders = []

        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price < fair_value - spread:
                buy_qty = min(-ask_volume, POSITION_LIMIT - position)
                if buy_qty > 0:
                    orders.append(Order(product, ask_price, buy_qty))
                    position += buy_qty

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > fair_value + spread:
                sell_qty = min(bid_volume, POSITION_LIMIT + position)
                if sell_qty > 0:
                    orders.append(Order(product, bid_price, -sell_qty))
                    position -= sell_qty

        # liquidation douce si trop en position
        if abs(position) > POSITION_LIMIT * 0.8:
            liquidation_price = fair_value + 1 if position > 0 else fair_value - 1
            liquidation_qty = min(abs(position), 5)
            orders.append(Order(product, liquidation_price, -liquidation_qty if position > 0 else liquidation_qty))

        return orders

    def handle_squid(self, product, order_depth, position, recent_prices):
        orders = []

        if len(recent_prices) >= 3:
            fair_value = mean(recent_prices)
        else:
            fair_value = FAIR_VALUES[product]

        deviation = max(1, int(fair_value * 0.02))  # spread dynamique 2%
        spread = max(2, deviation)

        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price < fair_value - spread:
                buy_qty = min(-ask_volume, POSITION_LIMIT - position)
                if buy_qty > 0:
                    orders.append(Order(product, ask_price, buy_qty))
                    position += buy_qty

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > fair_value + spread:
                sell_qty = min(bid_volume, POSITION_LIMIT + position)
                if sell_qty > 0:
                    orders.append(Order(product, bid_price, -sell_qty))
                    position -= sell_qty

        return orders
