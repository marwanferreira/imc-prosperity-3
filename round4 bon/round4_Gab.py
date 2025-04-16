from datamodel import Order, OrderDepth, TradingState, ConversionObservation
from typing import List, Dict
import jsonpickle
from math import sqrt

from round3 import Trader as BaseTrader, EPSILON

CONVERSION_LIMIT = 10
POSITION_LIMIT = 75
STORAGE_COST_PER_UNIT = 0.1

#La classe trader est adaptée pour le round 4, elle inclut la logique du code du round 3 :( version améliorée) 
# pour la tester : la remplacer entièrement (classe trader) dans le code round3
class Trader(BaseTrader): 
    def run(self, state: TradingState):
        result, conversions, memory = super().run(state)

        if isinstance(memory, str):
            try:
                memory = jsonpickle.decode(memory)
            except:
                memory = {}

        conversion_orders = {}
        macarons_orders, macarons_used = self.macarons_conversion_strategy(state, memory)
        conversions += macarons_used
        if macarons_orders:
            conversion_orders.update(macarons_orders)

        return result | conversion_orders, conversions, jsonpickle.encode(memory)

# nvlle méthode 
    def macarons_conversion_strategy(self, state: TradingState, memory: dict):
        observations = state.observations.conversionObservations
        symbol = "MAGNIFICENT_MACARONS"

        if symbol not in observations:
            return {}, 0

        conv: ConversionObservation = observations[symbol]
        position = state.position.get(symbol, 0)

        buy_price = conv.askPrice + conv.transportFees + conv.importTariff
        sell_price = conv.bidPrice - conv.transportFees - conv.exportTariff

        fair_value = (buy_price + sell_price) / 2
        spread = sell_price - buy_price

        memory.setdefault("macarons_fv_hist", []).append(fair_value)
        memory["macarons_fv_hist"] = memory["macarons_fv_hist"][-50:]

        avg_fv = sum(memory["macarons_fv_hist"]) / len(memory["macarons_fv_hist"])

        orders = []
        conversions_used = 0

# Logique du macarons

        # BUY LOGIC
        if buy_price < avg_fv * 0.985 and position < POSITION_LIMIT:
            buy_qty = min(CONVERSION_LIMIT - conversions_used, POSITION_LIMIT - position)
            if buy_qty > 0:
                orders.append(Order(symbol, int(buy_price), buy_qty))
                conversions_used += buy_qty
                position += buy_qty

        # SELL LOGIC
        if sell_price > avg_fv * 1.015 and position > -POSITION_LIMIT:
            sell_qty = min(CONVERSION_LIMIT - conversions_used, position + POSITION_LIMIT)
            if sell_qty > 0:
                orders.append(Order(symbol, int(sell_price), -sell_qty))
                conversions_used += sell_qty
                position -= sell_qty

        # STORAGE MITIGATION
        if conversions_used < CONVERSION_LIMIT and position > 0:
            if buy_price > sell_price:
                force_sell_qty = min(position, CONVERSION_LIMIT - conversions_used)
                if force_sell_qty > 0:
                    orders.append(Order(symbol, int(sell_price), -force_sell_qty))
                    conversions_used += force_sell_qty

        return {symbol: orders}, conversions_used
