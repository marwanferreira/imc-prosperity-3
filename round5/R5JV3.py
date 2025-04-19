from datamodel import Order, OrderDepth, TradingState, ConversionObservation
from typing import List, Dict
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
    "VOLCANIC_ROCK_VOUCHER_9500": 9500, "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000, "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}
VOL_WINDOW, MOMENTUM_WINDOW = 20, 10
EPSILON, MAX_TTE = 1e-6, 7
NUM_LEVELS, STOP_LOSS_TRIGGER = 3, 0.028
VOLATILITY_THRESHOLD = 5
BREAKOUT_WINDOW, BREAKOUT_BUFFER = 30, 1.0

# === MATH UTILITIES ===
def erf(x: float) -> float:
    a1,a2,a3,a4,a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p=0.3275911; sign=1 if x>=0 else -1; x=abs(x)
    t=1/(1+p*x)
    y=1-((((a5*t+a4)*t+a3)*t+a2)*t+a1)*t*exp(-x*x)
    return sign*y

def norm_cdf(x: float) -> float:
    return (1+erf(x/sqrt(2)))/2

def option_price_bs(S, K, T, sigma, r=0.0):
    if T<=0: return max(0.0, S-K)
    d1=(log(S/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T)+EPSILON)
    d2=d1-sigma*sqrt(T)
    return S*norm_cdf(d1)-K*exp(-r*T)*norm_cdf(d2)

def option_delta_bs(S, K, T, sigma, r=0.0):
    if T<=0: return 1.0 if S>K else 0.0
    d1=(log(S/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T)+EPSILON)
    return norm_cdf(d1)

class Trader:
    def run(self, state: TradingState):
        orders_out: Dict[str, List[Order]] = {}
        conversions = 0
        mem = jsonpickle.decode(state.traderData) if state.traderData else {}
        mem.setdefault("avg_costs", {}); mem.setdefault("hist_prices", {})
        avg_costs = mem["avg_costs"]; hist_prices = mem["hist_prices"]
        day = getattr(state, "day", 0); tte = max(0, MAX_TTE - day)

        # Update price history
        for prod, depth in state.order_depths.items():
            trades = state.market_trades.get(prod, [])
            hist_prices.setdefault(prod, [])
            for t in trades: hist_prices[prod].append(t.price)
            hist_prices[prod] = hist_prices[prod][-BREAKOUT_WINDOW:]

        # Signals
        fair, vol, mom = {}, {}, {}
        for prod, depth in state.order_depths.items():
            prices = [t.price for t in state.market_trades.get(prod, [])]
            fv = mean(prices) if prices else 100.0
            fair[prod] = fv
            v = pstdev(prices[-VOL_WINDOW:]) if len(prices)>=2 else 1.0
            vol[prod] = max(v, EPSILON)
            h = hist_prices.get(prod, [])
            mom[prod] = (h[-1]-h[0])/(h[0]+EPSILON) if len(h)>=2 else 0.0

        # Strategies
        for prod, depth in state.order_depths.items():
            pos = state.position.get(prod, 0); lim = POSITION_LIMITS.get(prod, 0)
            br = self.breakout_strategy(prod, depth, fair[prod], hist_prices.get(prod, []), pos, lim)
            mm = self.multi_level_market_make(prod, depth, fair[prod], vol[prod], mom[prod], pos, lim, avg_costs)
            orders_out[prod] = br + mm
            # Basket arbitrage
            if prod in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
                ba = self.basket_arbitrage(prod, BASKET1_COMPONENTS if prod=="PICNIC_BASKET1" else BASKET2_COMPONENTS, state, fair, vol)
                orders_out[prod] += ba
            # Vouchers
            if prod.startswith("VOLCANIC_ROCK_VOUCHER"):
                ov = self.voucher_market_make(prod, depth, fair.get("VOLCANIC_ROCK",0), vol.get("VOLCANIC_ROCK",1), mom.get("VOLCANIC_ROCK",0), pos, lim, tte)
                orders_out[prod] += ov

        # Conversions
        obs = state.observations
        if isinstance(obs, ConversionObservation):
            mac = "MAGNIFICENT_MACARONS"; pos = state.position.get(mac, 0)
            depth = state.order_depths.get(mac, OrderDepth())
            best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
            best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
            net_sell = obs.bidPrice - obs.transportFees - obs.exportTariff
            net_buy  = obs.askPrice + obs.transportFees + obs.importTariff
            if pos>0 and (best_bid is None or net_sell>best_bid+EPSILON):
                conversions = min(pos, CONVERSION_LIMITS[mac])
            if pos<0 and (best_ask is None or net_buy<best_ask-EPSILON):
                conversions = min(-pos, CONVERSION_LIMITS[mac])

        mem["avg_costs"] = avg_costs; mem["hist_prices"] = hist_prices
        return orders_out, conversions, jsonpickle.encode(mem)

    def breakout_strategy(self, prod, depth, fair, hist, pos, lim):
        orders: List[Order] = []
        if len(hist) < BREAKOUT_WINDOW: return orders
        hi, lo = max(hist), min(hist)
        bid, ask = self.get_best_bid(depth), self.get_best_ask(depth)
        if fair > hi + BREAKOUT_BUFFER and pos < lim and ask:
            orders.append(Order(prod, ask, 1))
        if fair < lo - BREAKOUT_BUFFER and pos > -lim and bid:
            orders.append(Order(prod, bid, -1))
        return orders

    def multi_level_market_make(self, prod, depth, fair, vol, mom, pos, lim, avg_costs):
        orders: List[Order] = []
        cur = pos
        # Stop-loss
        if prod in avg_costs and pos != 0:
            entry = avg_costs[prod]; pnl = (fair-entry)/(entry+EPSILON)
            if pos>0 and pnl<-STOP_LOSS_TRIGGER:
                q = max(1, pos//2); b = self.get_best_bid(depth)
                if b: orders.append(Order(prod, b, -q)); cur -= q
            if pos<0 and pnl>STOP_LOSS_TRIGGER:
                q = max(1, -pos//2); a = self.get_best_ask(depth)
                if a: orders.append(Order(prod, a, q)); cur += q
        # Spread
        sf, cm = (0.007,2.15) if vol<=VOLATILITY_THRESHOLD else (0.013,1.28)
        adj = -mom * 0.5 * sf * fair if abs(mom)>0.05 else 0
        base = max(1, int(vol + sf*fair)); spr = max(1, int(base+adj))
        spr += int(abs(cur)/lim*spr) if lim else 0; step = max(spr//NUM_LEVELS, 1)
        # Buys
        for lvl in range(1, NUM_LEVELS+1):
            tgt = fair - lvl*step
            for p, v in sorted(depth.sell_orders.items()):
                if p<=tgt and abs(cur)<lim:
                    q = min(int(abs(fair-p)/max(vol,EPSILON)*cm), lim-cur)
                    if q>0: orders.append(Order(prod, p, q)); cur += q
        # Sells
        for lvl in range(1, NUM_LEVELS+1):
            tgt = fair + lvl*step
            for p, v in sorted(depth.buy_orders.items(), reverse=True):
                if p>=tgt and abs(cur)<lim:
                    q = min(int(abs(fair-p)/max(vol,EPSILON)*cm), lim+cur)
                    if q>0: orders.append(Order(prod, p, -q)); cur -= q
        avg_costs[prod] = fair
        return orders

    def basket_arbitrage(self, basket, comps, state, fair, vol):
        orders: List[Order] = []
        depth = state.order_depths[basket]
        # compute synthetic
        cost_buy = sum(fair[c]*q for c, q in comps.items())
        cost_sell = cost_buy
        bid, ask = self.get_best_bid(depth), self.get_best_ask(depth)
        if bid and cost_sell > bid + EPSILON:
            orders.append(Order(basket, bid, -1))
        if ask and cost_buy < ask - EPSILON:
            orders.append(Order(basket, ask, 1))
        return orders

    def voucher_market_make(self, voucher, depth, rock_fv, rock_vol, rock_mom, pos, lim, tte):
        orders: List[Order] = []
        # Simple mid-price quote +/- small spread
        mid = option_price_bs(rock_fv, STRIKES[voucher], tte/252, rock_vol)
        bid_p, ask_p = int(mid*0.995), int(mid*1.005)
        orders.append(Order(voucher, bid_p, min(lim-pos, 1)))
        orders.append(Order(voucher, ask_p, -min(lim+pos, 1)))
        # Delta-hedge
        delta = option_delta_bs(rock_fv, STRIKES[voucher], tte/252, rock_vol)
        hedge_qty = int(delta * pos)
        if hedge_qty:
            # hedge in rock
            rock_bid = self.get_best_bid(depth)
            rock_ask = self.get_best_ask(depth)
            if hedge_qty>0 and rock_ask:
                orders.append(Order("VOLCANIC_ROCK", rock_ask, hedge_qty))
            if hedge_qty<0 and rock_bid:
                orders.append(Order("VOLCANIC_ROCK", rock_bid, hedge_qty))
        return orders

    def get_best_bid(self, depth: OrderDepth):
        return max(depth.buy_orders.keys()) if depth.buy_orders else None
    def get_best_ask(self, depth: OrderDepth):
        return min(depth.sell_orders.keys()) if depth.sell_orders else None