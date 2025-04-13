from datamodel import Order, OrderDepth, TradingState, Trade
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

MIN_SPREAD = {
    "RAINFOREST_RESIN": 1,
    "KELP": 2,
    "SQUID_INK": 8,
    "CROISSANTS": 1,
    "JAMS": 1,
    "DJEMBES": 1
}

ALPHA = 0.3  # Pour EMA (moyenne exponentielle)
HISTORY_LENGTH = 30

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

        fair_values = {}

        for product in state.order_depths:
            history_key = f"{product}_HISTORY"
            prices = trader_memory.get(history_key, [])
            trades: List[Trade] = state.market_trades.get(product, [])
            for trade in trades:
                prices.append(trade.price)
            prices = prices[-HISTORY_LENGTH:]
            trader_memory[history_key] = prices

            if len(prices) >= 3:
                ema = prices[0]
                for price in prices[1:]:
                    ema = ALPHA * price + (1 - ALPHA) * ema
                fair_values[product] = ema
            else:
                fair_values[product] = 100

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK", "CROISSANTS", "JAMS", "DJEMBES"]:
            orders = self.smart_market_maker(product, state.order_depths[product],
                                             fair_values[product], state.position.get(product, 0),
                                             POSITION_LIMITS[product], state.timestamp)
            if orders:
                result[product] = orders

        return result, conversions, jsonpickle.encode(trader_memory)

    def smart_market_maker(self, product, order_depth: OrderDepth, fair_value: float,
                            position: int, limit: int, timestamp: int) -> List[Order]:
        orders = []
        spread = max(MIN_SPREAD[product], int(fair_value * 0.01))

        # Volatilité adaptative : plus de prudence si haute variance
        if product == "SQUID_INK" and timestamp < 50000:
            spread += 5

        sorted_asks = sorted(order_depth.sell_orders.items())
        sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)

        # Buy logic
        for ask_price, ask_volume in sorted_asks:
            if ask_price < fair_value - spread:
                buy_qty = min(-ask_volume, limit - position)
                if buy_qty > 0:
                    orders.append(Order(product, ask_price, buy_qty))
                    position += buy_qty

        # Sell logic
        for bid_price, bid_volume in sorted_bids:
            if bid_price > fair_value + spread:
                sell_qty = min(bid_volume, limit + position)
                if sell_qty > 0:
                    orders.append(Order(product, bid_price, -sell_qty))
                    position -= sell_qty

        # Liquidation en fin de journée
        if timestamp > 99000 and position != 0:
            best_bid = sorted_bids[0][0] if sorted_bids else None
            best_ask = sorted_asks[0][0] if sorted_asks else None
            liquidation_price = best_bid if position > 0 else best_ask
            if liquidation_price:
                orders.append(Order(product, liquidation_price, -position))

        return orders

