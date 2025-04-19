from datamodel import Order, OrderDepth, TradingState, ConversionObservation, Listing
from typing import List, Dict
import jsonpickle
import logging
from math import log, sqrt, exp
from statistics import mean, pstdev

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONSTANTS & PARAMETERS ===
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
# Time windows
VOL_WINDOW = 20
MOM_WINDOW = 10
VWAP_WINDOW = 15
# Thresholds
VOL_THRESHOLD = 5
STOP_LOSS_PCT = 0.03
BREAKOUT_WINDOW = 30
BREAKOUT_BUFFER = 1.5
# Option parameters
MAX_TTE = 7  # days to expiry
EPS = 1e-6

# === UTILITY FUNCTIONS ===
def erf(x: float) -> float:
    # Approximation of the error function
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

def option_delta_bs(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0: return 1.0 if S > K else 0.0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*sqrt(T)+EPS)
    return norm_cdf(d1)

class Trader:
    def __init__(self):
        # Can initialize ML models or parameters here
        pass

    def run(self, state: TradingState):
        orders_out: Dict[str, List[Order]] = {}
        conversions = 0
        mem = jsonpickle.decode(state.traderData) if state.traderData else {}
        # Memory defaults
        mem.setdefault("avg_costs", {})
        mem.setdefault("price_hist", {})
        mem.setdefault("vwap_hist", {})
        mem.setdefault("cp_counts", {})
        avg_costs = mem["avg_costs"]
        price_hist = mem["price_hist"]
        vwap_hist = mem["vwap_hist"]
        cp_counts = mem["cp_counts"]

        # 1) Update counterparty stats
        for prod, trades in state.own_trades.items():
            for tr in trades:
                if tr.counter_party:
                    cp_counts[tr.counter_party] = cp_counts.get(tr.counter_party, 0) + 1

        # 2) Update price history & VWAP
        for sym, depth in state.order_depths.items():
            hist = price_hist.setdefault(sym, [])
            # record mid-price
            bid = max(depth.buy_orders.keys(), default=None)
            ask = min(depth.sell_orders.keys(), default=None)
            if bid and ask:
                mid = (bid + ask)/2
                hist.append(mid)
                price_hist[sym] = hist[-BREAKOUT_WINDOW*2:]
            # VWAP calc if trades exist
            trades = state.market_trades.get(sym, [])
            if trades:
                total_val = sum(t.price*t.quantity for t in trades)
                total_vol = sum(abs(t.quantity) for t in trades)
                vwap = total_val/total_vol if total_vol else mid
                vh = vwap_hist.setdefault(sym, [])
                vh.append(vwap)
                vwap_hist[sym] = vh[-VWAP_WINDOW:]

        # 3) Compute signals
        fair, vol, mom, vwap_sig = {}, {}, {}, {}
        for sym in state.order_depths:
            # fair = exponential moving average of mid-prices
            hist = price_hist.get(sym, [])
            fair[sym] = self._ema(hist, alpha=0.2)
            # volatility
            vol[sym] = pstdev(hist[-VOL_WINDOW:]) if len(hist)>=VOL_WINDOW else pstdev(hist) if len(hist)>=2 else 1.0
            # momentum
            mom[sym] = (hist[-1] - hist[0])/(hist[0]+EPS) if len(hist)>=MOM_WINDOW else 0.0
            # VWAP divergence
            vh = vwap_hist.get(sym, [])
            vwap_sig[sym] = (fair[sym] - (vh[-1] if vh else fair[sym]))/(fair[sym]+EPS)

        # 4) Strategies per product
        day = getattr(state, "day", 0)
        tte = max(0, MAX_TTE - day)
        for sym, depth in state.order_depths.items():
            pos = state.position.get(sym, 0)
            lim = POSITION_LIMITS.get(sym, 0)
            orders = []
            # A) Breakout strategy
            orders += self.breakout_strategy(sym, fair, price_hist, pos, lim)
            # B) Mean-reversion around VWAP
            orders += self.vwap_reversion(sym, fair[sym], vwap_sig[sym], depth, pos, lim)
            # C) Multi-level market maker
            orders += self.multi_level_mm(sym, fair[sym], vol[sym], mom[sym], pos, lim, depth, avg_costs, cp_counts)
            # D) Basket arbitrage
            if sym in ["PICNIC_BASKET1","PICNIC_BASKET2"]:
                comp = BASKET1_COMPONENTS if sym.endswith("1") else BASKET2_COMPONENTS
                orders += self.basket_arbitrage(sym, comp, state, fair, vol)
            # E) Voucher quoting + BS hedging
            if sym.startswith("VOLCANIC_ROCK_VOUCHER"):
                orders += self.voucher_strategy(sym, fair.get("VOLCANIC_ROCK",0), vol.get("VOLCANIC_ROCK",1), pos, lim, tte, depth)
            orders_out[sym] = orders

        # 5) Conversion decision
        obs = state.observations
        if isinstance(obs, ConversionObservation):
            conv_qty = self.conversion_logic(obs, state.position.get("MAGNIFICENT_MACARONS",0), state.order_depths.get("MAGNIFICENT_MACARONS", OrderDepth()))
            conversions = conv_qty

        # 6) Persist memory
        mem.update({"avg_costs": avg_costs, "price_hist": price_hist, "vwap_hist": vwap_hist, "cp_counts": cp_counts})
        return orders_out, conversions, jsonpickle.encode(mem)

    # === Helper methods ===
    def _ema(self, data: List[float], alpha: float) -> float:
        if not data: return 100.0
        ema = data[0]
        for x in data[1:]: ema = alpha*x + (1-alpha)*ema
        return ema

    def breakout_strategy(self, sym, fair_map, hist, pos, lim):
        orders = []
        h = hist.get(sym, [])[-BREAKOUT_WINDOW:]
        if len(h)<BREAKOUT_WINDOW: return []
        hi, lo = max(h), min(h)
        fv = fair_map[sym]
        bid, ask = self.get_best_bid, self.get_best_ask
        if fv > hi + BREAKOUT_BUFFER and pos < lim:
            orders.append(Order(sym, ask(depth), 1))
        if fv < lo - BREAKOUT_BUFFER and pos > -lim:
            orders.append(Order(sym, bid(depth), -1))
        return orders

    def vwap_reversion(self, sym, fv, vwap_diff, depth, pos, lim):
        orders = []
        threshold = 0.01
        bid, ask = self.get_best_bid(depth), self.get_best_ask(depth)
        if vwap_diff > threshold and pos > -lim:
            orders.append(Order(sym, bid, -1))  # sell into premium
        if vwap_diff < -threshold and pos < lim:
            orders.append(Order(sym, ask, 1))   # buy into discount
        return orders

    def multi_level_mm(self, sym, fv, vol, mom, pos, lim, depth, avg_costs, cp_counts):
        # (Expands existing multi-level code with additional risk caps, dealer inventory prioritization...)
        ...  # implement full code as before

    def basket_arbitrage(self, basket, comps, state, fair, vol):
        # (As before)
        ...

    def voucher_strategy(self, voucher, rock_fv, rock_vol, pos, lim, tte, depth):
        # (BS pricing, dynamic spread, partial hedge)
        ...

    def conversion_logic(self, obs: ConversionObservation, pos: int, depth: OrderDepth) -> int:
        # use sugarPrice and sunlightIndex to adjust thresholds
        bid, ask = self.get_best_bid(depth), self.get_best_ask(depth)
        net_sell = obs.bidPrice - obs.transportFees - obs.exportTariff
        net_buy  = obs.askPrice + obs.transportFees + obs.importTariff
        conv = 0
        mood = obs.sunlightIndex / (obs.sugarPrice+EPS)
        if pos > 0 and net_sell > (bid or 0) + 0.5*mood:
            conv = min(pos, CONVERSION_LIMITS["MAGNIFICENT_MACARONS"])
        if pos < 0 and net_buy < (ask or 1e9) - 0.5*mood:
            conv = min(-pos, CONVERSION_LIMITS["MAGNIFICENT_MACARONS"])
        return conv

    def get_best_bid(self, depth: OrderDepth): return max(depth.buy_orders.keys()) if depth.buy_orders else None
    def get_best_ask(self, depth: OrderDepth): return min(depth.sell_orders.keys()) if depth.sell_orders else None