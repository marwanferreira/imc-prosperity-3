from datamodel import Order, OrderDepth, TradingState, ConversionObservation, Trade
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
EPSILON, MAX_TIME_TO_EXPIRY = 1e-6, 7
NUM_LEVELS, STOP_LOSS_TRIGGER = 3, 0.028
CP_EXPLOIT_THRESHOLD = 5
CP_PRICE_DEVIATION = 0.01  # 1%

# === MATH UTILITIES ===
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

# === Black-Scholes for vouchers ===
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

class Trader:
    def run(self, state: TradingState):
        orders_out: Dict[str, List[Order]] = {}
        conversions = 0
        memory = jsonpickle.decode(state.traderData) if state.traderData else {}
        memory.setdefault("avg_costs", {})
        memory.setdefault("cp_stats", {})
        avg_costs = memory["avg_costs"]
        cp_stats = memory["cp_stats"]

        current_day = getattr(state, "day", 0)
        time_to_expiry = max(0, MAX_TIME_TO_EXPIRY - current_day)
        memory["time_to_expiry"] = time_to_expiry

        # Update counterparty stats
        for prod, trades in state.own_trades.items():
            cp_stats.setdefault(prod, {})
            for t in trades:
                cp = getattr(t, 'counter_party', t.seller or t.buyer)
                if not cp: continue
                stats = cp_stats[prod].get(cp, {"count":0, "avg_price":0.0})
                count = stats["count"] + 1
                avg_p = (stats["avg_price"]*stats["count"] + t.price)/count
                cp_stats[prod][cp] = {"count":count, "avg_price":avg_p}
        memory["cp_stats"] = cp_stats

        # Compute fair, vol, momentum
        fair_values, volatilities, momentums = {}, {}, {}
        for prod, depth in state.order_depths.items():
            trades = state.market_trades.get(prod, [])
            ema_key, hist_key = f"{prod}_EMA", f"{prod}_HIST"
            ema = memory.get(ema_key, mean([t.price for t in trades]) if trades else 100.0)
            hist = memory.get(hist_key, [])
            for t in trades:
                ema = 0.2*t.price + 0.8*ema
                hist.append(t.price)
            hist = hist[-(VOL_WINDOW+10):]
            memory[ema_key], memory[hist_key] = ema, hist
            recent = mean(hist) if hist else ema
            fair = 0.7*ema + 0.3*recent
            fair_values[prod] = fair
            vol = pstdev(hist[-VOL_WINDOW:]) if len(hist)>=2 else 1.0
            volatilities[prod] = max(vol, EPSILON)
            mom = (hist[-1]-hist[-MOMENTUM_WINDOW])/(hist[-MOMENTUM_WINDOW]+EPSILON) if len(hist)>=MOMENTUM_WINDOW else 0.0
            momentums[prod] = mom

        # Main strategies
        for prod in ["RAINFOREST_RESIN","KELP","SQUID_INK","CROISSANTS","JAMS","DJEMBES","VOLCANIC_ROCK"]:
            if prod in state.order_depths:
                orders_out[prod] = self.multi_level_market_make(
                    prod, state.order_depths[prod], fair_values[prod], volatilities[prod], momentums[prod], state.position.get(prod,0), POSITION_LIMITS[prod], avg_costs, memory)

        # Basket arbitrage
        for basket, comps in [("PICNIC_BASKET1",BASKET1_COMPONENTS),("PICNIC_BASKET2",BASKET2_COMPONENTS)]:
            if basket in state.order_depths:
                orders_out[basket] = self.basket_arbitrage(basket, comps, state, fair_values, volatilities)

        # Voucher market making
        rock_fv = fair_values.get("VOLCANIC_ROCK",10000)
        rock_vol = volatilities.get("VOLCANIC_ROCK",100)
        for v in STRIKES:
            if v in state.order_depths:
                orders_out[v] = self.voucher_market_make(v, state.order_depths[v], rock_fv, rock_vol, momentums.get("VOLCANIC_ROCK",0.0), state.position.get(v,0), POSITION_LIMITS[v], time_to_expiry, memory)

        # Conversions
        mac = "MAGNIFICENT_MACARONS"
        if isinstance(state.observations, ConversionObservation):
            pos = state.position.get(mac,0)
            limit_conv = CONVERSION_LIMITS[mac]
            depth_mac = state.order_depths.get(mac, OrderDepth())
            best_bid = max(depth_mac.buy_orders.keys()) if depth_mac.buy_orders else None
            best_ask = min(depth_mac.sell_orders.keys()) if depth_mac.sell_orders else None
            obs = state.observations
            net_sell = obs.bidPrice - obs.transportFees - obs.exportTariff
            net_buy = obs.askPrice + obs.transportFees + obs.importTariff
            if pos>0 and (best_bid is None or net_sell>best_bid+EPSILON):
                conversions = min(pos, limit_conv)
            elif pos<0 and (best_ask is None or net_buy<best_ask-EPSILON):
                conversions = min(abs(pos), limit_conv)

        memory["avg_costs"] = avg_costs
        return orders_out, conversions, jsonpickle.encode(memory)

    def multi_level_market_make(self, product: str, depth: OrderDepth,
                                 fair: float, vol: float, mom: float,
                                 pos: int, lim: int,
                                 avg_costs: Dict[str,float], mem: dict) -> List[Order]:
        orders: List[Order] = []
        # Exploit counterparties
        cp_info = mem.get("cp_stats", {}).get(product, {})
        for cp, stats in cp_info.items():
            if stats["count"]>=CP_EXPLOIT_THRESHOLD:
                # overpriced seller => buy
                if stats["avg_price"]>fair*(1+CP_PRICE_DEVIATION):
                    p = self.get_best_ask(depth)
                    if p:
                        q = min(lim-pos, stats["count"])
                        if q>0:
                            orders.append(Order(product,p,q)); pos+=q
                # underpriced buyer => sell
                elif stats["avg_price"]<fair*(1-CP_PRICE_DEVIATION):
                    p = self.get_best_bid(depth)
                    if p:
                        q = min(lim+pos, stats["count"])
                        if q>0:
                            orders.append(Order(product,p,-q)); pos-=q
        # === Stop-loss ===
        if product in avg_costs and pos!=0:
            entry = avg_costs[product]
            pnl = (fair-entry)/(entry+EPSILON)
            if pos>0 and pnl< -STOP_LOSS_TRIGGER:
                b=self.get_best_bid(depth)
                if b: orders.append(Order(product,b,-max(1,pos//2))); pos-=max(1,pos//2)
            elif pos<0 and pnl>STOP_LOSS_TRIGGER:
                a=self.get_best_ask(depth)
                if a: orders.append(Order(product,a,max(1,abs(pos)//2))); pos+=max(1,abs(pos)//2)
        # === Multi-level spread ===
        sf, cm = (0.007,2.15) if vol<=VOLATILITY_THRESHOLD else (0.013,1.28)
        adj = -mom*0.5*sf*fair if abs(mom)>0.05 else 0
        base = max(1,int(vol+sf*fair)); spr=max(1,base+int(adj))
        spr+=int(abs(pos)/lim*spr)
        step=max(spr//NUM_LEVELS,1)
        # buy levels
        cur=pos
        for lvl in range(1,NUM_LEVELS+1):
            tgt=fair-lvl*step
            for p,v in sorted(depth.sell_orders.items()):
                if p<=tgt:
                    c=abs(fair-p)/max(vol,EPSILON)
                    q=min(int(c*cm), lim-cur)
                    if q>0:
                        orders.append(Order(product,p,q)); avg_costs[product]=self.update_avg_cost(avg_costs.get(product,0),cur,p,q); cur+=q
                if abs(cur)>=lim: break
            if abs(cur)>=lim: break
        # sell levels
        for lvl in range(1,NUM_LEVELS+1):
            tgt=fair+lvl*step
            for p,v in sorted(depth.buy_orders.items(),reverse=True):
                if p>=tgt:
                    c=abs(fair-p)/max(vol,EPSILON)
                    q=min(int(c*cm), lim+cur)
                    if q>0:
                        orders.append(Order(product,p,-q)); avg_costs[product]=self.update_avg_cost(avg_costs.get(product,0),cur,p,-q); cur-=q
                if abs(cur)>=lim: break
            if abs(cur)>=lim: break
        mem.setdefault("positions_override",{})[product]=cur
        return orders

    def basket_arbitrage(self, basket: str, comps: Dict[str,int],
                         state: TradingState, fair: Dict[str,float], vol: Dict[str,float]) -> List[Order]:
        orders: List[Order] = []
        # Round4 logic: buy basket if undervalued, sell if overvalued
        # Calculate theoretical basket price
        theo = sum(fair[p]*qty for p,qty in comps.items())
        depth = state.order_depths[basket]
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_ask and best_ask < theo:
            qty = min(sum(state.position.get(p,0)//qty for p,qty in comps.items()), comps[next(iter(comps))])
            orders.append(Order(basket,best_ask,qty))
        if best_bid and best_bid > theo:
            qty = min(sum(abs(state.position.get(p,0)//qty) for p,qty in comps.items()), comps[next(iter(comps))])
            orders.append(Order(basket,best_bid,-qty))
        return orders

    def voucher_market_make(self, voucher: str, depth: OrderDepth,
                             rock_fv: float, rock_vol: float,
                             rock_mom: float, pos: int,
                             lim: int, tte: int, mem: dict) -> List[Order]:
        orders: List[Order] = []
        K = STRIKES[voucher]
        sigma = rock_vol
        T = tte/252
        fair_opt = option_price_bs(rock_fv, K, T, sigma)
        delta = option_delta_bs(rock_fv, K, T, sigma)
        # hedge delta if needed
        # Place quotes around fair_opt
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_ask and best_ask < fair_opt:
            qty = min(lim-pos, depth.sell_orders[best_ask])
            orders.append(Order(voucher,best_ask,qty)); pos+=qty
        if best_bid and best_bid > fair_opt:
            qty = min(lim+pos, depth.buy_orders[best_bid])
            orders.append(Order(voucher,best_bid,-qty)); pos-=qty
        return orders

    def get_best_bid(self, depth: OrderDepth): return max(depth.buy_orders.keys()) if depth.buy_orders else None
    def get_best_ask(self, depth: OrderDepth): return min(depth.sell_orders.keys()) if depth.sell_orders else None
    def update_avg_cost(self, old_avg: float, old_size: int, price: float, qty: int) -> float:
        new_size = old_size + qty
        if new_size == 0: return 0.0
        return (old_avg*old_size + price*qty)/new_size
