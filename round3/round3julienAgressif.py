from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import jsonpickle
from math import log, sqrt, exp
from statistics import mean, pstdev

POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 50,
    "JAMS": 50,
    "DJEMBES": 50,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
}

BASKET1_COMPONENTS = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
BASKET2_COMPONENTS = {"CROISSANTS": 4, "JAMS": 2}

STRIKES = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}

VOL_WINDOW = 10
MOMENTUM_WINDOW = 5
EPSILON = 1e-6
VOLATILITY_THRESHOLD = 3
BASE_ALPHA = 0.9
MAX_TIME_TO_EXPIRY = 7
NUM_LEVELS = 5 
STOP_LOSS_TRIGGER = 0.04

def erf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0/(1.0+p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)
    return sign * y

def norm_cdf(x: float) -> float:
    return (1.0 + erf(x/sqrt(2.0))) / 2.0

def option_price_bs(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T) + EPSILON)
    d2 = d1 - sigma * sqrt(T)
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)

def option_delta_bs(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T) + EPSILON)
    return norm_cdf(d1)

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

        if "avg_costs" not in memory:
            memory["avg_costs"] = {}
        avg_costs = memory["avg_costs"]

        current_day = getattr(state, "day", 0)
        time_to_expiry = max(0, MAX_TIME_TO_EXPIRY - current_day)
        memory["time_to_expiry"] = time_to_expiry

        fair_values = {}
        volatilities = {}
        regimes = {}
        momentums = {}
        for product in state.order_depths:
            trades = state.market_trades.get(product, [])
            ema_key = f"{product}_EMA"
            hist_key = f"{product}_HIST"
            ema = memory.get(ema_key, 100)
            hist = memory.get(hist_key, [])
            for trade in trades:
                ema = 0.2 * trade.price + 0.8 * ema
                hist.append(trade.price)
            hist = hist[-(VOL_WINDOW + 10):]
            memory[ema_key], memory[hist_key] = ema, hist
            recent_mean = mean(hist) if hist else ema
            fair_val = 0.7 * ema + 0.3 * recent_mean
            fair_values[product] = fair_val
            if len(hist) > 1:
                vol = pstdev(hist[-VOL_WINDOW:])
            else:
                vol = 1.0
            vol = max(vol, EPSILON)
            volatilities[product] = vol
            regimes[product] = "volatile" if vol > VOLATILITY_THRESHOLD else "stable"
            if len(hist) >= MOMENTUM_WINDOW:
                mom = (hist[-1] - hist[-MOMENTUM_WINDOW]) / (hist[-MOMENTUM_WINDOW] + EPSILON)
            else:
                mom = 0
            momentums[product] = mom

        for product in [
            "RAINFOREST_RESIN", "KELP", "SQUID_INK",
            "CROISSANTS", "JAMS", "DJEMBES", "VOLCANIC_ROCK"
        ]:
            if product in state.order_depths:
                pos = state.position.get(product, 0)
                limit = POSITION_LIMITS[product]
                orders = self.multi_level_market_make(
                    product,
                    state.order_depths[product],
                    fair_values[product],
                    volatilities[product],
                    regimes[product],
                    momentums[product],
                    pos,
                    limit,
                    avg_costs,
                    memory
                )
                if orders:
                    result[product] = orders

        for basket, comps in [
            ("PICNIC_BASKET1", BASKET1_COMPONENTS),
            ("PICNIC_BASKET2", BASKET2_COMPONENTS)
        ]:
            if basket in state.order_depths:
                orders = self.basket_arbitrage(
                    basket,
                    comps,
                    state,
                    fair_values,
                    volatilities
                )
                if orders:
                    result[basket] = orders

        rock_fv = fair_values.get("VOLCANIC_ROCK", 10000)
        rock_vol = volatilities.get("VOLCANIC_ROCK", 100)
        for voucher in STRIKES:
            if voucher in state.order_depths:
                pos_v = state.position.get(voucher, 0)
                limit_v = POSITION_LIMITS[voucher]
                orders_voucher = self.voucher_market_make(
                    voucher,
                    state.order_depths[voucher],
                    rock_fv,
                    rock_vol,
                    momentums["VOLCANIC_ROCK"],
                    pos_v,
                    limit_v,
                    time_to_expiry,
                    memory
                )
                if orders_voucher:
                    result[voucher] = orders_voucher

        memory["avg_costs"] = avg_costs
        return result, conversions, jsonpickle.encode(memory)

    def multi_level_market_make(self, product: str, order_depth: OrderDepth,
                                fair_value: float, volatility: float, regime: str,
                                momentum: float, position: int, limit: int,
                                avg_costs: Dict[str, float], memory: dict) -> List[Order]:
        orders = []
        if position != 0 and product in avg_costs and avg_costs[product] != 0:
            current_price = fair_value
            entry_price = avg_costs[product]
            pnl_ratio = (current_price - entry_price) / (entry_price + EPSILON)
            if position > 0 and pnl_ratio < -STOP_LOSS_TRIGGER:
                partial_close = max(1, abs(position) // 2)
                best_bid = self.get_best_bid(order_depth)
                if best_bid:
                    orders.append(Order(product, best_bid, -partial_close))
                    position -= partial_close
            elif position < 0 and pnl_ratio > STOP_LOSS_TRIGGER:
                partial_close = max(1, abs(position) // 2)
                best_ask = self.get_best_ask(order_depth)
                if best_ask:
                    orders.append(Order(product, best_ask, partial_close))
                    position += partial_close
            memory.setdefault("positions_override", {})[product] = position

        if regime == "stable":
            spread_factor = 0.005
            confidence_mult = 3
        else:
            spread_factor = 0.009
            confidence_mult = 1.8

        if momentum > 0.05:
            momentum_spread_adj = - (abs(momentum) * 0.5 * spread_factor * fair_value)
        elif momentum < -0.05:
            momentum_spread_adj = + (abs(momentum) * 0.5 * spread_factor * fair_value)
        else:
            momentum_spread_adj = 0

        base_spread = max(1, int(volatility + fair_value * spread_factor))
        final_spread = max(1, int(base_spread + momentum_spread_adj))
        pos_ratio = abs(position) / float(limit) if limit else 0
        final_spread += int(pos_ratio * final_spread)
        step = max(final_spread // (NUM_LEVELS if NUM_LEVELS > 0 else 1), 1)
        current_position = position

        for i in range(1, NUM_LEVELS + 1):
            buy_level_price = fair_value - i * step
            for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                if ask_price <= buy_level_price:
                    confidence = abs(fair_value - ask_price) / max(volatility, EPSILON)
                    qty = min(int(confidence * confidence_mult), limit - current_position)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        new_size = current_position + qty
                        avg_costs[product] = self.update_avg_cost(avg_costs.get(product, 0), current_position, ask_price, qty)
                        current_position = new_size
                    if abs(current_position) >= limit:
                        break
            if abs(current_position) >= limit:
                break

        for i in range(1, NUM_LEVELS + 1):
            sell_level_price = fair_value + i * step
            for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price >= sell_level_price:
                    confidence = abs(fair_value - bid_price) / max(volatility, EPSILON)
                    qty = min(int(confidence * confidence_mult), limit + current_position)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        new_size = current_position - qty
                        avg_costs[product] = self.update_avg_cost(avg_costs.get(product, 0), current_position, bid_price, -qty)
                        current_position = new_size
                    if abs(current_position) >= limit:
                        break
            if abs(current_position) >= limit:
                break

        memory.setdefault("positions_override", {})[product] = current_position
        return orders

    def breakout_strategy(self, product: str, state: TradingState,
                          fair_value: float, momentum: float,
                          local_high_low: tuple, position: int, limit: int,
                          avg_costs: Dict[str, float], memory: dict) -> List[Order]:
        orders = []
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
                if abs(position) >= limit:
                    break
        for ask_price, ask_volume in sorted(depth.sell_orders.items()):
            if ask_price < comp_value - threshold:
                confidence = abs(comp_value - ask_price) / comp_vol
                qty = min(int(confidence * 2), limit - position)
                if qty > 0:
                    orders.append(Order(basket_name, ask_price, qty))
                    position += qty
                if abs(position) >= limit:
                    break
        return orders

    def voucher_market_make(self, voucher_name: str,
                            voucher_depth: OrderDepth,
                            rock_fair_value: float,
                            rock_volatility: float,
                            rock_momentum: float,
                            position_voucher: int,
                            limit_voucher: int,
                            time_to_expiry: int,
                            memory: dict) -> List[Order]:
        orders = []
        strike = STRIKES[voucher_name]
        T_year = time_to_expiry / 252.0
        sigma = (rock_volatility / rock_fair_value) if rock_fair_value > 0 else 0.2
        price_bs = option_price_bs(rock_fair_value, strike, T_year, sigma)
        intrinsic = max(0, rock_fair_value - strike)
        time_val = BASE_ALPHA * rock_volatility * sqrt(time_to_expiry + 1)
        voucher_fv = (price_bs + intrinsic + time_val) / 2.0

        base_spread = max(2, int(rock_volatility * 0.3))
        if abs(rock_momentum) > 0.05:
            base_spread += int(rock_volatility * abs(rock_momentum))
        pos_ratio = abs(position_voucher) / float(limit_voucher) if limit_voucher else 0
        final_spread = base_spread + int(pos_ratio * base_spread)

        max_levels = 3
        for level in range(1, max_levels + 1):
            buy_trigger = voucher_fv - final_spread * level
            sell_trigger = voucher_fv + final_spread * level
            for ask_price, ask_volume in sorted(voucher_depth.sell_orders.items()):
                if ask_price < buy_trigger:
                    factor = level
                    qty = min(ask_volume, factor * 2, limit_voucher - position_voucher)
                    if qty > 0:
                        orders.append(Order(voucher_name, ask_price, qty))
                        position_voucher += qty
                        hedge_ratio = min(0.3 + 0.7 * (intrinsic / max(rock_fair_value, 1)), 1)
                        if intrinsic > 0:
                            self.delta_hedge("VOLCANIC_ROCK", qty, memory, is_buy_voucher=True, ratio=hedge_ratio)
                if abs(position_voucher) >= limit_voucher:
                    break
            for bid_price, bid_volume in sorted(voucher_depth.buy_orders.items(), reverse=True):
                if bid_price > sell_trigger:
                    factor = level
                    qty = min(bid_volume, factor * 2, limit_voucher + position_voucher)
                    if qty > 0:
                        orders.append(Order(voucher_name, bid_price, -qty))
                        position_voucher -= qty
                        hedge_ratio = min(0.3 + 0.7 * (intrinsic / max(rock_fair_value, 1)), 1)
                        if intrinsic > 0:
                            self.delta_hedge("VOLCANIC_ROCK", qty, memory, is_buy_voucher=False, ratio=hedge_ratio)
                if abs(position_voucher) >= limit_voucher:
                    break
        return orders

    def delta_hedge(self, rock_name: str, qty_voucher: int, memory: dict,
                    is_buy_voucher: bool, ratio: float = 0.5):
        hedge_orders = memory.get("hedge_orders", [])
        hedge_qty = int(qty_voucher * ratio)
        if hedge_qty <= 0:
            return
        if is_buy_voucher:
            hedge_orders.append((rock_name, -hedge_qty))
        else:
            hedge_orders.append((rock_name, hedge_qty))
        memory["hedge_orders"] = hedge_orders

    def update_avg_cost(self, old_avg: float, old_size: int, trade_price: float, trade_qty: int) -> float:
        new_size = old_size + trade_qty
        if new_size == 0:
            return 0.0
        old_value = old_avg * old_size
        trade_value = trade_price * trade_qty
        return (old_value + trade_value) / new_size

    def get_best_bid(self, order_depth: OrderDepth):
        if not order_depth.buy_orders:
            return None
        return max(order_depth.buy_orders.keys())

    def get_best_ask(self, order_depth: OrderDepth):
        if not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys())