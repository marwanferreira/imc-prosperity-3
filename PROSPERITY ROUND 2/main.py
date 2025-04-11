import json
from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState
from collections import deque
import math
import numpy as np

class Trader:
    """
    Bot avancé pour Round 2 de Prosperity 3.
    Principales fonctionnalités :
    1) Market making sur RAINFOREST_RESIN (autour d'une fair value fixe ~10k).
    2) Mean reversion sur KELP et SQUID_INK (EMA dynamique + seuils).
    3) Arbitrage sur PICNIC_BASKET1 et PICNIC_BASKET2 via leur valeur intrinsèque
       (6*C + 3*J + D) / (4*C + 2*J) respectivement.
    4) Gestion de risque dynamique :
       - Positions surveillées pour respecter les limites.
       - Liquidation automatique en fin de round ou si position trop importante.
    5) Journalisation et logs basiques (PNL estimé, positions, etc.).

    Retour attendu par run(): (orders: Dict[str, List[Order]], conversion: Dict, trader_data: str)
    Où:
        - orders: dictionnaire {produit -> liste d'ordres}
        - conversion: un dictionnaire si tu veux faire des conversions internes (souvent vide)
        - trader_data: string JSON-serializable, contenant info debug ou stats.
    """

    def __init__(self):
        # Limites de positions
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAM": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        # Positions courantes
        self.positions = {p: 0 for p in self.position_limits}

        # Price history & EMA for KELP and SQUID_INK
        self.price_history = {
            "KELP": deque(maxlen=20),
            "SQUID_INK": deque(maxlen=20)
        }
        self.ema = {"KELP": None, "SQUID_INK": None}
        self.alpha = 0.2  # Facteur de lissage EMA

        # Seuils de détection pour mean reversion
        self.threshold = {
            "KELP": 2,       # Ecart par rapport à l'EMA
            "SQUID_INK": 3   # Ecart plus grand car plus volatile
        }

        # Paramètres d'arbitrage pour Picnic Baskets
        # Diff (fair_value - basket_mid) au-delà duquel on prend position
        self.arb_threshold_pb1 = 5
        self.arb_threshold_pb2 = 4

        # Log interne
        self.log_messages = []

        # Liquidation automatique
        # - On suppose qu'on veut liquider TOUTES les positions au timestamp > 980_000 (par ex.),
        #   ou si un drawdown potentiel trop fort.
        self.end_of_round = 980_000  # hypothétique fin de round. Ajuster si besoin.

        # Suivi PNL naïf (approximation, juste pour logs)
        self.pnl_estimate = 0.0
        # On stocke le prix d'entrée moyen / la quantité pour calculer un PNL approximatif.
        self.average_price = {p: 0.0 for p in self.position_limits}
        self.total_traded = {p: 0 for p in self.position_limits}

    def log(self, message: str):
        """Stocke le message dans un buffer pour debug."""
        self.log_messages.append(message)

    def print_logs(self):
        """Méthode d'affichage éventuel. Dans Prosperity on a pas toujours la console,
        mais on peut tout de même print."""
        for m in self.log_messages:
            print(m)
        self.log_messages.clear()

    def get_mid_price(self, order_depth: OrderDepth):
        """Retourne le mid price si possible, sinon None."""
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def update_positions(self, state: TradingState):
        """Met à jour self.positions avec les valeurs du state."""
        for product in self.position_limits:
            self.positions[product] = state.position.get(product, 0)

    def update_pnl_estimate(self, state: TradingState):
        """Estime un PNL approximatif en marquant à la fair value (mid price) nos positions."""
        running_pnl = 0.0
        for product in self.position_limits:
            pos = self.positions[product]
            od = state.order_depths.get(product, None)
            if od and pos != 0:
                mp = self.get_mid_price(od)
                if mp is not None:
                    # Approximatif : pos * mp
                    running_pnl += pos * mp
        self.pnl_estimate = running_pnl  # c'est un mark-to-market partiel
        # On pourrait rajouter le realized PnL en conservant un historique des trades.

    def update_ema(self, product: str, mid_price: float):
        """Met à jour l'EMA avec un alpha donné."""
        if self.ema[product] is None:
            self.ema[product] = mid_price
        else:
            self.ema[product] = self.alpha * mid_price + (1 - self.alpha) * self.ema[product]

    def fair_value_picnic1(self, state: TradingState):
        """Calcule la fair value de PICNIC_BASKET1 = 6*C + 3*J + 1*Djembe"""
        od_c = state.order_depths.get("CROISSANTS", None)
        od_j = state.order_depths.get("JAM", None)
        od_d = state.order_depths.get("DJEMBE", None)
        if od_c and od_j and od_d:
            c_mid = self.get_mid_price(od_c)
            j_mid = self.get_mid_price(od_j)
            d_mid = self.get_mid_price(od_d)
            if c_mid and j_mid and d_mid:
                return 6 * c_mid + 3 * j_mid + d_mid
        return None

    def fair_value_picnic2(self, state: TradingState):
        """Calcule la fair value de PICNIC_BASKET2 = 4*C + 2*J"""
        od_c = state.order_depths.get("CROISSANTS", None)
        od_j = state.order_depths.get("JAM", None)
        if od_c and od_j:
            c_mid = self.get_mid_price(od_c)
            j_mid = self.get_mid_price(od_j)
            if c_mid and j_mid:
                return 4 * c_mid + 2 * j_mid
        return None

    def place_order_if_profitable(self, result_dict, product: str, price: float, volume: int):
        """Fonction utilitaire pour placer un ordre si volume != 0."""
        if volume == 0:
            return
        # On le place directement.
        result_dict[product].append(Order(product, price, volume))
        self.log(f"Placing order on {product}: price={price}, vol={volume}")

    def handle_liquidation(self, state: TradingState, result: Dict[str, List[Order]]):
        """Liquidation automatique :
        - Si on approche de la fin du round (timestamp > self.end_of_round).
        - Peut aussi être appelée si on veut tout simplement forcer la position à 0.
        """
        if state.timestamp < self.end_of_round:
            return  # Pas encore en phase de liquidation

        # On tente de ramener toutes les positions à 0
        for product, pos in self.positions.items():
            if pos == 0:
                continue
            od = state.order_depths.get(product, None)
            if not od or (not od.buy_orders and not od.sell_orders):
                continue  # Pas d'info pour liquider

            if pos > 0:
                # Pour liquider, on vend (ordre négatif)
                best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
                if best_bid is not None:
                    volume_possible = min(od.buy_orders[best_bid], pos)
                    self.place_order_if_profitable(result, product, best_bid, -volume_possible)
            else:
                # pos < 0, on veut acheter pour se couvrir
                best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
                if best_ask is not None:
                    volume_possible = min(-od.sell_orders[best_ask], -pos)
                    self.place_order_if_profitable(result, product, best_ask, volume_possible)

    def run(self, state: TradingState):
        """Point d'entrée. Retourne un triple (orders, conversion, traderData) pour coller à l'API.
        - orders (Dict[str, List[Order]]): Les ordres à exécuter.
        - conversion (Dict): Peut rester vide.
        - traderData (str): Une chaîne JSON.
        """
        # 1) Mettre à jour positions / PnL estimé
        self.update_positions(state)
        self.update_pnl_estimate(state)

        # 2) Préparer la structure de résultat
        result = {p: [] for p in self.position_limits}

        # 3) RAINFOREST_RESIN : market making autour de 10k
        r_depth = state.order_depths.get("RAINFOREST_RESIN", None)
        if r_depth:
            # Valeur supposée ~10k
            fair_resin = 10000
            rpos = self.positions["RAINFOREST_RESIN"]

            # On prend d'abord les opportunités de marché existantes
            # (acheter si vend < fair_resin, vendre si acheteur > fair_resin)
            sorted_sells = sorted(r_depth.sell_orders.items(), key=lambda x: x[0])
            for sell_price, sell_vol in sorted_sells:
                if sell_price < fair_resin and rpos < self.position_limits["RAINFOREST_RESIN"]:
                    vol = min(-sell_vol, self.position_limits["RAINFOREST_RESIN"] - rpos)
                    if vol > 0:
                        self.place_order_if_profitable(result, "RAINFOREST_RESIN", sell_price, vol)
                        rpos += vol

            sorted_buys = sorted(r_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            for buy_price, buy_vol in sorted_buys:
                if buy_price > fair_resin and rpos > -self.position_limits["RAINFOREST_RESIN"]:
                    vol = min(buy_vol, self.position_limits["RAINFOREST_RESIN"] + rpos)
                    if vol > 0:
                        self.place_order_if_profitable(result, "RAINFOREST_RESIN", buy_price, -vol)
                        rpos -= vol

            # Puis on place nos propres quotes autour de fair_resin
            # Adjusté en fonction du rpos.
            # Spread = 1 Seashell de chaque côté

            # Côté acheteur
            if rpos < self.position_limits["RAINFOREST_RESIN"]:
                buy_vol = self.position_limits["RAINFOREST_RESIN"] - rpos
                buy_price = fair_resin - 1
                # Si on est déjà long, on veut être moins agressif, disons fair_resin - 2
                if rpos > 0:
                    buy_price = fair_resin - 2
                self.place_order_if_profitable(result, "RAINFOREST_RESIN", buy_price, buy_vol)

            # Côté vendeur
            if rpos > -self.position_limits["RAINFOREST_RESIN"]:
                sell_vol = self.position_limits["RAINFOREST_RESIN"] + rpos
                sell_price = fair_resin + 1
                if rpos < 0:
                    sell_price = fair_resin + 2
                self.place_order_if_profitable(result, "RAINFOREST_RESIN", sell_price, -sell_vol)

        # 4) KELP & SQUID_INK : mean reversion
        for product in ["KELP", "SQUID_INK"]:
            depth = state.order_depths.get(product, None)
            if not depth:
                continue
            midp = self.get_mid_price(depth)
            if midp:
                # Mettre à jour EMA
                self.update_ema(product, midp)
                self.price_history[product].append(midp)

            pos = self.positions[product]
            # Exploiter si le marché a un mispricing par rapport à l'EMA.
            if self.ema[product] is not None and midp is not None:
                diff = midp - self.ema[product]
                # Condition d'achat: diff < -threshold => le prix actuel est trop bas
                if diff < -self.threshold[product] and pos < self.position_limits[product]:
                    best_ask = None
                    best_ask_vol = 0
                    if depth.sell_orders:
                        best_ask = min(depth.sell_orders.keys())
                        best_ask_vol = -depth.sell_orders[best_ask]
                    if best_ask is not None:
                        volume_can_buy = self.position_limits[product] - pos
                        volume = min(best_ask_vol, volume_can_buy)
                        if volume > 0:
                            self.place_order_if_profitable(result, product, best_ask, volume)
                # Condition de vente: diff > threshold => le prix actuel est trop haut
                elif diff > self.threshold[product] and pos > -self.position_limits[product]:
                    best_bid = None
                    best_bid_vol = 0
                    if depth.buy_orders:
                        best_bid = max(depth.buy_orders.keys())
                        best_bid_vol = depth.buy_orders[best_bid]
                    if best_bid is not None:
                        volume_can_sell = self.position_limits[product] + pos
                        volume = min(best_bid_vol, volume_can_sell)
                        if volume > 0:
                            self.place_order_if_profitable(result, product, best_bid, -volume)

            # Market making autour de l'EMA
            if self.ema[product] is not None:
                fair = self.ema[product]
                ppos = pos
                # Acheteur
                if ppos < self.position_limits[product]:
                    buy_price = fair - (1 if product == "KELP" else 2)  # Squid plus large
                    vol_buy = self.position_limits[product] - ppos
                    self.place_order_if_profitable(result, product, buy_price, vol_buy)
                # Vendeur
                if ppos > -self.position_limits[product]:
                    sell_price = fair + (1 if product == "KELP" else 2)
                    vol_sell = self.position_limits[product] + ppos
                    self.place_order_if_profitable(result, product, sell_price, -vol_sell)

        # 5) Picnic Basket 1 & 2 : Arbitrage statique

        # PB1
        pb1_depth = state.order_depths.get("PICNIC_BASKET1", None)
        pos_pb1 = self.positions["PICNIC_BASKET1"]
        fair_pb1 = self.fair_value_picnic1(state)
        if pb1_depth and fair_pb1:
            mid_pb1 = self.get_mid_price(pb1_depth)
            if mid_pb1 is not None:
                spread = fair_pb1 - mid_pb1
                # Si le basket est sous-évalué => spread > threshold => on achète le basket
                if spread > self.arb_threshold_pb1 and pos_pb1 < self.position_limits["PICNIC_BASKET1"]:
                    best_ask = None
                    best_ask_vol = 0
                    if pb1_depth.sell_orders:
                        best_ask = min(pb1_depth.sell_orders.keys())
                        best_ask_vol = -pb1_depth.sell_orders[best_ask]
                    if best_ask is not None and best_ask_vol > 0:
                        volume_can_buy = self.position_limits["PICNIC_BASKET1"] - pos_pb1
                        vol = min(best_ask_vol, volume_can_buy)
                        if vol > 0:
                            self.place_order_if_profitable(result, "PICNIC_BASKET1", best_ask, vol)

                # Si le basket est sur-évalué => spread < -threshold => on vend le basket
                if spread < -self.arb_threshold_pb1 and pos_pb1 > -self.position_limits["PICNIC_BASKET1"]:
                    best_bid = None
                    best_bid_vol = 0
                    if pb1_depth.buy_orders:
                        best_bid = max(pb1_depth.buy_orders.keys())
                        best_bid_vol = pb1_depth.buy_orders[best_bid]
                    if best_bid is not None and best_bid_vol > 0:
                        volume_can_sell = self.position_limits["PICNIC_BASKET1"] + pos_pb1
                        vol = min(best_bid_vol, volume_can_sell)
                        if vol > 0:
                            self.place_order_if_profitable(result, "PICNIC_BASKET1", best_bid, -vol)

        # PB2
        pb2_depth = state.order_depths.get("PICNIC_BASKET2", None)
        pos_pb2 = self.positions["PICNIC_BASKET2"]
        fair_pb2 = self.fair_value_picnic2(state)
        if pb2_depth and fair_pb2:
            mid_pb2 = self.get_mid_price(pb2_depth)
            if mid_pb2 is not None:
                spread = fair_pb2 - mid_pb2
                # Sous-évalué => on achète
                if spread > self.arb_threshold_pb2 and pos_pb2 < self.position_limits["PICNIC_BASKET2"]:
                    best_ask = None
                    best_ask_vol = 0
                    if pb2_depth.sell_orders:
                        best_ask = min(pb2_depth.sell_orders.keys())
                        best_ask_vol = -pb2_depth.sell_orders[best_ask]
                    if best_ask is not None and best_ask_vol > 0:
                        volume_can_buy = self.position_limits["PICNIC_BASKET2"] - pos_pb2
                        vol = min(best_ask_vol, volume_can_buy)
                        if vol > 0:
                            self.place_order_if_profitable(result, "PICNIC_BASKET2", best_ask, vol)

                # Sur-évalué => on vend
                if spread < -self.arb_threshold_pb2 and pos_pb2 > -self.position_limits["PICNIC_BASKET2"]:
                    best_bid = None
                    best_bid_vol = 0
                    if pb2_depth.buy_orders:
                        best_bid = max(pb2_depth.buy_orders.keys())
                        best_bid_vol = pb2_depth.buy_orders[best_bid]
                    if best_bid is not None and best_bid_vol > 0:
                        volume_can_sell = self.position_limits["PICNIC_BASKET2"] + pos_pb2
                        vol = min(best_bid_vol, volume_can_sell)
                        if vol > 0:
                            self.place_order_if_profitable(result, "PICNIC_BASKET2", best_bid, -vol)

        # 6) Liquidation automatique en fin de round
        self.handle_liquidation(state, result)

        # 7) Logging
        self.log(f"Timestamp: {state.timestamp}, PnL approx: {self.pnl_estimate}")
        for prod, pos in self.positions.items():
            self.log(f" - {prod} pos: {pos}")

        self.print_logs()

        # On prépare un JSON (string) pour traderData
        trader_data = {
            "pnl_estimate": self.pnl_estimate,
            "positions": self.positions
        }
        trader_data_str = json.dumps(trader_data)

        # On renvoie 3 valeurs pour coller à l'API :
        #  - Les ordres
        #  - Un dict conversion vide
        #  - Le traderData sous forme de string JSON.
        return result, {}, trader_data_str
