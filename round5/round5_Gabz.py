from datamodel import Order, OrderDepth, TradingState, ConversionObservation, Trade
from typing import List, Dict, Tuple
import jsonpickle
from math import log, sqrt, exp
from statistics import mean, pstdev

# === CONSTANTS ===
POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "CROISSANTS": 50, "JAMS": 50, "DJEMBES": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75,
}
CONVERSION_LIMITS = {"MAGNIFICENT_MACARONS": 10}
BASKET1_COMPONENTS = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
BASKET2_COMPONENTS = {"CROISSANTS": 4, "JAMS": 2}
STRIKES = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}
# === PARAMETERS ===
VOL_WINDOW = 20
MOM_WINDOW = 10
VWAP_WINDOW = 15
BREAKOUT_WINDOW = 30
BREAKOUT_BUFFER = 1.5
STOP_LOSS_PCT = 0.03
MAX_TTE = 7
EPS = 1e-6
SIGNAL_BOOST = 12

# === UTILS ===
def erf(x: float) -> float:
    a1,a2,a3,a4,a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p=0.3275911; sign = 1 if x >= 0 else -1; x = abs(x)
    t = 1/(1 + p*x)
    y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)
    return sign*y

def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x/sqrt(2)))

def option_price_bs(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0: return max(0.0, S-K)
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*sqrt(T)+EPS)
    d2 = d1 - sigma*sqrt(T)
    return S*norm_cdf(d1) - K*exp(-r*T)*norm_cdf(d2)

class SignalEngine:
    PAIR_SIGNAL_MAP = {
        ("RAINFOREST_RESIN", "Penelope", "Charlie"): "long",
        ("RAINFOREST_RESIN", "Gary", "Charlie"): "long",
        ("RAINFOREST_RESIN", "Paris", "Charlie"): "long",
        ("PICNIC_BASKET1", "Camilla", "Penelope"): "long",
    }
    def detect_signals(self, trades: Dict[str, List[Trade]]) -> Dict[str, str]:
        signals = {}
        for sym, trade_list in trades.items():
            for t in trade_list[-5:]:
                key = (sym, t.seller, t.buyer)
                if key in self.PAIR_SIGNAL_MAP:
                    signals[sym] = self.PAIR_SIGNAL_MAP[key]
        return signals

class Trader:
    def __init__(self):
        self.signal_engine = SignalEngine()

    def run(self, state: TradingState):
        orders_out: Dict[str, List[Order]] = {}
        conversions = 0
        mem = jsonpickle.decode(state.traderData) if state.traderData else {}

        mem.setdefault("avg_costs", {})
        mem.setdefault("price_hist", {})
        mem.setdefault("vwap_hist", {})
        avg_costs = mem["avg_costs"]
        price_hist = mem["price_hist"]
        vwap_hist = mem["vwap_hist"]

        signals = self.signal_engine.detect_signals(state.market_trades)

        for sym, depth in state.order_depths.items():
            hist = price_hist.setdefault(sym, [])
            bid = max(depth.buy_orders.keys(), default=None)
            ask = min(depth.sell_orders.keys(), default=None)
            if bid and ask:
                mid = (bid + ask)/2
                hist.append(mid)
                price_hist[sym] = hist[-BREAKOUT_WINDOW*2:]

            trades = state.market_trades.get(sym, [])
            if trades:
                total_val = sum(t.price*t.quantity for t in trades)
                total_vol = sum(abs(t.quantity) for t in trades)
                vwap = total_val/total_vol if total_vol else mid
                vh = vwap_hist.setdefault(sym, [])
                vh.append(vwap)
                vwap_hist[sym] = vh[-VWAP_WINDOW:]

        fair, vol, mom, vwap_sig = {}, {}, {}, {}
        for sym in state.order_depths:
            hist = price_hist.get(sym, [])
            fair[sym] = self._ema(hist, alpha=0.2)
            vol[sym] = pstdev(hist[-VOL_WINDOW:]) if len(hist)>=VOL_WINDOW else 1.0
            mom[sym] = (hist[-1] - hist[0])/(hist[0]+EPS) if len(hist)>=MOM_WINDOW else 0.0
            vh = vwap_hist.get(sym, [])
            vwap_sig[sym] = (fair[sym] - (vh[-1] if vh else fair[sym]))/(fair[sym]+EPS)

        day = getattr(state, "day", 0)
        tte = max(0, MAX_TTE - day)
        for sym, depth in state.order_depths.items():
            pos = state.position.get(sym, 0)
            lim = POSITION_LIMITS.get(sym, 0)
            orders = []

            # Signal boost
            if sym in signals and signals[sym] == "long":
                fair[sym] += SIGNAL_BOOST
                best_ask = self.get_best_ask(depth)
                if best_ask and pos < lim:
                    orders.append(Order(sym, best_ask, 3))

            orders += self.breakout(sym, fair[sym], price_hist.get(sym, []), pos, lim, depth)
            orders += self.vwap_reversion(sym, fair[sym], vwap_sig[sym], depth, pos, lim)
            orders += self.market_making(sym, fair[sym], vol[sym], mom[sym], depth, pos, lim)
            if sym in ["PICNIC_BASKET1","PICNIC_BASKET2"]:
                comp = BASKET1_COMPONENTS if sym.endswith("1") else BASKET2_COMPONENTS
                orders += self.basket_arbitrage(sym, comp, state, fair, vol)
            if sym.startswith("VOLCANIC_ROCK_VOUCHER"):
                orders += self.voucher_pricing(sym, fair.get("VOLCANIC_ROCK",0), vol.get("VOLCANIC_ROCK",1), pos, lim, tte, depth)
            orders_out[sym] = orders

        obs = state.observations
        if isinstance(obs, ConversionObservation):
            pos = state.position.get("MAGNIFICENT_MACARONS", 0)
            depth = state.order_depths.get("MAGNIFICENT_MACARONS", OrderDepth())
            conversions = self.macaron_conversion(obs, pos, depth)

        mem.update({"avg_costs": avg_costs, "price_hist": price_hist, "vwap_hist": vwap_hist})
        return orders_out, conversions, jsonpickle.encode(mem)

    def breakout(self, sym, fv, hist, pos, lim, depth):
        orders = []
        h = hist[-BREAKOUT_WINDOW:]
        if len(h)<BREAKOUT_WINDOW: return []
        hi, lo = max(h), min(h)
        bid = self.get_best_bid(depth)
        ask = self.get_best_ask(depth)
        if fv > hi + BREAKOUT_BUFFER and pos < lim:
            orders.append(Order(sym, ask, 1))
        if fv < lo - BREAKOUT_BUFFER and pos > -lim:
            orders.append(Order(sym, bid, -1))
        return orders

    def vwap_reversion(self, sym, fv, vwap_diff, depth, pos, lim):
        orders = []
        threshold = 0.01
        bid = self.get_best_bid(depth)
        ask = self.get_best_ask(depth)
        if vwap_diff > threshold and pos > -lim:
            orders.append(Order(sym, bid, -1))
        if vwap_diff < -threshold and pos < lim:
            orders.append(Order(sym, ask, 1))
        return orders

    def market_making(self, sym, fv, vol, mom, depth, pos, lim):
        orders = []
        spread = max(1, int(vol + fv * 0.01))
        step = max(spread // 2, 1)
        for i in range(1, 3):
            buy_price = fv - i * step
            sell_price = fv + i * step
            if pos < lim:
                orders.append(Order(sym, int(buy_price), 1))
            if pos > -lim:
                orders.append(Order(sym, int(sell_price), -1))
        return orders

    def basket_arbitrage(self, basket, comps, state, fair, vol):
        orders = []
        depth = state.order_depths[basket]
        pos = state.position.get(basket, 0)
        lim = POSITION_LIMITS[basket]
        comp_val = sum(fair[p]*q for p,q in comps.items())
        comp_vol = sum(vol[p]*q for p,q in comps.items())
        threshold = max(3, int(comp_vol * 0.5))
        for bid, _ in sorted(depth.buy_orders.items(), reverse=True):
            if bid > comp_val + threshold and pos > -lim:
                orders.append(Order(basket, bid, -1))
        for ask, _ in sorted(depth.sell_orders.items()):
            if ask < comp_val - threshold and pos < lim:
                orders.append(Order(basket, ask, 1))
        return orders

    def voucher_pricing(self, voucher, rock_fv, rock_vol, pos, lim, tte, depth):
        orders = []
        strike = STRIKES[voucher]
        T = tte/252
        sigma = rock_vol / rock_fv if rock_fv else 0.2
        price = option_price_bs(rock_fv, strike, T, sigma)
        base_spread = max(2, int(rock_vol * 0.3))
        for ask, _ in sorted(depth.sell_orders.items()):
            if ask < price - base_spread and pos < lim:
                orders.append(Order(voucher, ask, 1))
        for bid, _ in sorted(depth.buy_orders.items(), reverse=True):
            if bid > price + base_spread and pos > -lim:
                orders.append(Order(voucher, bid, -1))
        return orders

    def macaron_conversion(self, obs: ConversionObservation, pos: int, depth: OrderDepth) -> int:
        bid = self.get_best_bid(depth)
        ask = self.get_best_ask(depth)
        net_sell = obs.bidPrice - obs.transportFees - obs.exportTariff
        net_buy  = obs.askPrice + obs.transportFees + obs.importTariff
        mood = obs.sunlightIndex / (obs.sugarPrice+EPS)
        if pos > 0 and net_sell > (bid or 0) + 0.5*mood:
            return min(pos, CONVERSION_LIMITS["MAGNIFICENT_MACARONS"])
        if pos < 0 and net_buy < (ask or 1e9) - 0.5*mood:
            return min(-pos, CONVERSION_LIMITS["MAGNIFICENT_MACARONS"])
        return 0

    def _ema(self, data: List[float], alpha: float) -> float:
        if not data: return 100.0
        ema = data[0]
        for x in data[1:]:
            ema = alpha * x + (1 - alpha) * ema
        return ema

    def get_best_bid(self, depth: OrderDepth):
        return max(depth.buy_orders.keys()) if depth.buy_orders else None

    def get_best_ask(self, depth: OrderDepth):
        return min(depth.sell_orders.keys()) if depth.sell_orders else None
