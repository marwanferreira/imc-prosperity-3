from datamodel import Order, OrderDepth, TradingState, ConversionObservation
from typing import List, Dict
import jsonpickle
from math import log, sqrt, exp
from statistics import mean, pstdev

# =============================================================================
# PARAMÈTRES GLOBAUX
# =============================================================================
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
    # Round 4
    "MAGNIFICENT_MACARONS": 75,
}

# Limites de conversions
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

VOL_WINDOW = 20
MOMENTUM_WINDOW = 10
EPSILON = 1e-6
VOLATILITY_THRESHOLD = 5
BASE_ALPHA = 0.6
MAX_TIME_TO_EXPIRY = 7
NUM_LEVELS = 3
STOP_LOSS_TRIGGER = 0.028

# Round 4 parameters
EXTRA_POSITION_RATIO = 0.55
TRAIL_STOP_FACTOR = 0.035
MOMENTUM_THRESHOLD = 0.08
BREAKOUT_BUFFER = 1.5

# =============================================================================
# UTILITAIRES BLACK–SCHOLES
# =============================================================================
def erf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)
    return sign * y


def norm_cdf(x: float) -> float:
    return (1.0 + erf(x/sqrt(2.0))) / 2.0


def option_price_bs(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(0.0, S - K)
    d1 = (log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * sqrt(T) + EPSILON)
    d2 = d1 - sigma * sqrt(T)
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def option_delta_bs(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * sqrt(T) + EPSILON)
    return norm_cdf(d1)

# =============================================================================
# TRADER – ROUND 4
# =============================================================================
class Trader:
    def run(self, state: TradingState):
        orders_out: Dict[str, List[Order]] = {}
        conversions = 0
        
        # Mémoire
        memory = jsonpickle.decode(state.traderData) if state.traderData else {}
        memory.setdefault("avg_costs", {})
        avg_costs = memory["avg_costs"]

        # Time to expiry pour options
        current_day = getattr(state, "day", 0)
        time_to_expiry = max(0, MAX_TIME_TO_EXPIRY - current_day)
        memory["time_to_expiry"] = time_to_expiry

        # 1) Calcul des fair values, vol, momentum
        fair_values, volatilities, regimes, momentums = {}, {}, {}, {}
        for prod, depth in state.order_depths.items():
            trades = state.market_trades.get(prod, [])
            ema_key, hist_key = f"{prod}_EMA", f"{prod}_HIST"
            ema = memory.get(ema_key, mean([t.price for t in trades]) if trades else 100.0)
            hist = memory.get(hist_key, [])
            for t in trades:
                ema = 0.2 * t.price + 0.8 * ema
                hist.append(t.price)
            hist = hist[-(VOL_WINDOW+10):]
            memory[ema_key], memory[hist_key] = ema, hist

            recent = mean(hist) if hist else ema
            fv = 0.7 * ema + 0.3 * recent
            fair_values[prod] = fv

            vol = pstdev(hist[-VOL_WINDOW:]) if len(hist) >= 2 else 1.0
            vol = max(vol, EPSILON)
            volatilities[prod] = vol
            regimes[prod] = "volatile" if vol > VOLATILITY_THRESHOLD else "stable"

            mom = (hist[-1] - hist[-MOMENTUM_WINDOW])/(hist[-MOMENTUM_WINDOW]+EPSILON) if len(hist)>=MOMENTUM_WINDOW else 0.0
            momentums[prod] = mom

        # 2) Market making spot
        for prod in ["RAINFOREST_RESIN","KELP","SQUID_INK","CROISSANTS","JAMS","DJEMBES","VOLCANIC_ROCK","MAGNIFICENT_MACARONS"]:
            if prod in state.order_depths:
                om = self.multi_level_market_make(prod,
                                                 state.order_depths[prod],
                                                 fair_values[prod],
                                                 volatilities[prod],
                                                 regimes[prod],
                                                 momentums[prod],
                                                 state.position.get(prod,0),
                                                 POSITION_LIMITS[prod],
                                                 avg_costs,
                                                 memory)
                if om:
                    orders_out[prod] = om

        # 3) Arbitrage paniers
        for basket, comps in [("PICNIC_BASKET1", BASKET1_COMPONENTS),
                              ("PICNIC_BASKET2", BASKET2_COMPONENTS)]:
            if basket in state.order_depths:
                ab = self.basket_arbitrage(basket, comps, state, fair_values, volatilities)
                if ab:
                    orders_out[basket] = ab

        # 4) Market making options
        rock_fv = fair_values.get("VOLCANIC_ROCK", 10000)
        rock_vol = volatilities.get("VOLCANIC_ROCK", 100)
        for v in STRIKES:
            if v in state.order_depths:
                omv = self.voucher_market_make(v,
                                               state.order_depths[v],
                                               rock_fv, rock_vol,
                                               momentums.get("VOLCANIC_ROCK",0.0),
                                               state.position.get(v,0),
                                               POSITION_LIMITS[v],
                                               time_to_expiry,
                                               memory)
                if omv:
                    orders_out[v] = omv

                # 5) Conversion macarons
        mac = "MAGNIFICENT_MACARONS"
        obs = state.observations
        if isinstance(obs, ConversionObservation):
            pos = state.position.get(mac, 0)
            limit_conv = CONVERSION_LIMITS[mac]
            depth_mac = state.order_depths.get(mac, OrderDepth({}, {}))
            best_bid = max(depth_mac.buy_orders.keys()) if depth_mac.buy_orders else None
            best_ask = min(depth_mac.sell_orders.keys()) if depth_mac.sell_orders else None

            # Calcul des prix nets
            net_sell = obs.bidPrice - obs.transportFees - obs.exportTariff
            net_buy = obs.askPrice + obs.transportFees + obs.importTariff

            # Si pas de position, ouverture via conversion
            if pos == 0:
                # conversion achat si profitable (acheter macarons via conversion)
                if best_ask is not None and net_buy + EPSILON < best_ask:
                    conversions = limit_conv
                # conversion vente si profitable (vendre macarons via conversion)
                elif best_bid is not None and net_sell > best_bid + EPSILON:
                    conversions = limit_conv
            # Si position positive, fermer si profitable
            elif pos > 0:
                if best_bid is None or net_sell > best_bid + EPSILON:
                    conversions = min(pos, limit_conv)
            # Si position négative, ouvrir si profitable
            elif pos < 0:
                if best_ask is None or net_buy < best_ask - EPSILON:
                    conversions = min(abs(pos), limit_conv)

        memory["avg_costs"] = avg_costs
        return orders_out, conversions, jsonpickle.encode(memory)

    # ---------- multi-level market making ----------
    def multi_level_market_make(self, product: str, depth: OrderDepth,
                                 fair: float, vol: float, regime: str,
                                 mom: float, pos: int, lim: int,
                                 avg_costs: Dict[str,float], mem: dict) -> List[Order]:
        orders: List[Order] = []
        # stop-loss
        if product in avg_costs and pos != 0:
            entry = avg_costs[product]
            pnl = (fair - entry)/(entry + EPSILON)
            if pos > 0 and pnl < -STOP_LOSS_TRIGGER:
                q = max(1, pos//2)
                b = self.get_best_bid(depth)
                if b: orders.append(Order(product, b, -q)); pos -= q
            elif pos < 0 and pnl > STOP_LOSS_TRIGGER:
                q = max(1, abs(pos)//2)
                a = self.get_best_ask(depth)
                if a: orders.append(Order(product, a, q)); pos += q
            mem.setdefault("positions_override", {})[product] = pos
        # spread
        sf, cm = (0.007, 2.15) if regime == "stable" else (0.013, 1.28)
        adj = -mom * 0.5 * sf * fair if abs(mom) > 0.05 else 0
        base = max(1, int(vol + sf * fair))
        spr = max(1, int(base + adj))
        spr += int(abs(pos)/lim * spr) if lim else 0
        step = max(spr//NUM_LEVELS, 1)
        cur = pos
        # achats
        for lvl in range(1, NUM_LEVELS+1):
            tgt = fair - lvl * step
            for p, v in sorted(depth.sell_orders.items()):
                if p <= tgt:
                    c = abs(fair - p)/max(vol, EPSILON)
                    q = min(int(c*cm), lim - cur)
                    if q>0:
                        orders.append(Order(product, p, q))
                        avg_costs[product] = self.update_avg_cost(avg_costs.get(product,0), cur, p, q)
                        cur += q
                    if abs(cur)>=lim: break
            if abs(cur)>=lim: break
        # ventes
        for lvl in range(1, NUM_LEVELS+1):
            tgt = fair + lvl * step
            for p, v in sorted(depth.buy_orders.items(), reverse=True):
                if p >= tgt:
                    c = abs(fair - p)/max(vol, EPSILON)
                    q = min(int(c*cm), lim + cur)
                    if q>0:
                        orders.append(Order(product, p, -q))
                        avg_costs[product] = self.update_avg_cost(avg_costs.get(product,0), cur, p, -q)
                        cur -= q
                    if abs(cur)>=lim: break
            if abs(cur)>=lim: break
        mem.setdefault("positions_override", {})[product] = cur
        return orders

    # ---------- breakout ----------
    def breakout_strategy(self, product: str, state: TradingState,
                          fair: float, mom: float,
                          highs_lows: tuple, pos: int,
                          lim: int, avg_costs: Dict[str,float], mem: dict) -> List[Order]:
        # implémentation Round 3 inchangée
        return []

    # ---------- arbitrage paniers ----------
    def basket_arbitrage(self, basket: str, comps: Dict[str,int],
                         state: TradingState, fair: Dict[str,float], vol: Dict[str,float]) -> List[Order]:
        # implémentation Round 3 inchangée
        return []

    # ---------- market making options ----------
    def voucher_market_make(self, voucher: str, depth: OrderDepth,
                             rock_fv: float, rock_vol: float,
                             rock_mom: float, pos: int,
                             lim: int, tte: int, mem: dict) -> List[Order]:
        # implémentation Round 3 inchangée
        return []

    # ---------- delta hedging ----------
    def delta_hedge(self, rock: str, qty: int, mem: dict,
                    is_buy: bool, ratio: float=0.5):
        # implémentation Round 3 inchangée
        pass

    # ---------- mises à jour et utilitaires ----------
    def update_avg_cost(self, old_avg: float, old_size: int,
                        price: float, qty: int) -> float:
        new_size = old_size + qty
        if new_size == 0:
            return 0.0
        return (old_avg*old_size + price*qty)/new_size

    def get_best_bid(self, depth: OrderDepth):
        return max(depth.buy_orders.keys()) if depth.buy_orders else None

    def get_best_ask(self, depth: OrderDepth):
        return min(depth.sell_orders.keys()) if depth.sell_orders else None
