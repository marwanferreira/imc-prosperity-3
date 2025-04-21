from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
from numpy import mean, std
import string

class Product:
    def __init__(self, name, sell_order_history, buy_order_history, current_position):
        # Name
        self.name = name

        # Sell order history
        self.sell_order_history = sell_order_history
        self.sell_order_average = get_average(self.sell_order_history)

        # Buy order history
        self.buy_order_history = buy_order_history
        self.buy_order_average = get_average(self.buy_order_history)

        # Mid Price
        self.average_mid_price = (self.sell_order_average + self.sell_order_average) / 2

        # Position information
        self.position = current_position
        self.position_limit = get_position_limits()[name]

        # Default buy and sell thresholds
        self.default_offset = self.calculate_offset(10, 3)
        self.current_offset = self.default_offset
        self.acceptable_buy_price = self.average_mid_price - self.default_offset
        self.acceptable_sell_price = self.average_mid_price + self.default_offset
    
    def calculate_offset(self, range, divisor=3):
        if len(self.sell_order_history) == 0:
            return 0

        index_one = 0
        index_two = range
        if len(self.sell_order_history) < (range + 1):
            index_two = len(self.sell_order_history) - 1
        
        sell_offset = (self.sell_order_history[index_one] - self.sell_order_history[-index_two]) / divisor
        if sell_offset < 0:
            sell_offset *= -1
        
        return sell_offset

    def set_buy_price_offset(self, new_offset):
        self.acceptable_buy_price -= self.current_offset
        self.acceptable_buy_price += new_offset

        self.current_offset = new_offset

    def set_sell_price_offset(self, new_offset):
        self.acceptable_sell_price -= self.current_offset
        self.acceptable_sell_price += new_offset

        self.current_offset = new_offset

    def set_acceptable_buy_price(self, new_price):
        self.acceptable_buy_price = new_price - self.current_offset
    
    def set_acceptable_sell_price(self, new_price):
        self.acceptable_sell_price = new_price + self.current_offset

class Macaron(Product):
    def __init__(self, name, sell_order_history, buy_order_history, current_position, observation_info_history, current_observation_info):
        #super().__init__(name, sell_order_history, buy_order_history, current_position)
        self.name = name

        self.sell_order_history = sell_order_history
        self.buy_order_history = buy_order_history
        
        self.sell_order_average = 0
        if len(sell_order_history) > 0:
            self.sell_order_average = mean(sell_order_history)
        
        self.buy_order_average = 0
        if len(buy_order_history) > 0:
            self.buy_order_average = mean(buy_order_history)

        self.historical_average_mid_price = (self.sell_order_average + self.sell_order_average) / 2

        self.position = current_position

        self.observation_info_history = observation_info_history
        
        self.historical_ask_price_mean = 0
        if len(observation_info_history["askPrice"]) > 0:
            self.historical_ask_price_mean = mean(observation_info_history["askPrice"])

        self.historical_bid_price_mean = 0
        if len(observation_info_history["bidPrice"]) > 0:
            self.historical_bid_price_mean = mean(observation_info_history["bidPrice"])

        self.historical_export_tariff_mean = 0
        if len(observation_info_history["exportTariff"]) > 0:
            self.historical_export_tariff_mean = mean(observation_info_history["exportTariff"])

        self.historical_import_tariff_mean = 0
        if len(observation_info_history["importTariff"]) > 0:
            self.historical_import_tariff_mean = mean(observation_info_history["importTariff"])

        self.historical_sugar_price_mean = 0
        if len(observation_info_history["sugarPrice"]) > 0:
            self.historical_sugar_price_mean = mean(observation_info_history["sugarPrice"])

        self.historical_sunlight_mean = 0
        if len(observation_info_history["sunlightIndex"]) > 0:
            self.historical_sunlight_mean = mean(observation_info_history["sunlightIndex"])

        self.historical_transport_fees_mean = 0
        if len(observation_info_history["transportFees"]) > 0:
            self.historical_transport_fees_mean = mean(observation_info_history["transportFees"])


        self.historical_ask_price_std = 0
        if len(observation_info_history["askPrice"]) > 0:
            self.historical_ask_price_std = std(observation_info_history["askPrice"])
        
        self.historical_bid_price_std = 0
        if len(observation_info_history["bidPrice"]) > 0:
            self.historical_bid_price_std = std(observation_info_history["bidPrice"])
        
        self.historical_export_tariff_std = 0
        if len(observation_info_history["exportTariff"]) > 0:
            self.historical_export_tariff_std = std(observation_info_history["exportTariff"])
        
        self.historical_import_tariff_std = 0
        if len(observation_info_history["importTariff"]) > 0:
            self.historical_import_tariff_std = std(observation_info_history["importTariff"])
        
        self.historical_sugar_price_std = 0
        if len(observation_info_history["sugarPrice"]) > 0:
            self.historical_sugar_price_std = std(observation_info_history["sugarPrice"])
        
        self.historical_sunlight_std = 0
        if len(observation_info_history["sunlightIndex"]) > 0:
            self.historical_sunlight_std = std(observation_info_history["sunlightIndex"])
        
        self.historical_transport_fees_std = 0
        if len(observation_info_history["transportFees"]) > 0:
            self.historical_transport_fees_std = std(observation_info_history["transportFees"])
        
        self.current_average_mid_price = (self.historical_ask_price_mean + self.historical_bid_price_mean) / 2

        self.current_observation_info = current_observation_info

        self.ask_price = current_observation_info.askPrice
        self.bid_price = current_observation_info.bidPrice
        self.export_tariff = current_observation_info.exportTariff
        self.import_tariff = current_observation_info.importTariff
        self.sugar_price = current_observation_info.sugarPrice
        self.sunlight = current_observation_info.sunlightIndex
        self.transport_fees = current_observation_info.transportFees

        # maybe comment out
        """ self.normalized_ask_price = 0
        if self.historical_ask_price_std != 0:
            self.normalized_ask_price = (self.ask_price - self.historical_ask_price_mean) / self.historical_ask_price_std
        
        self.normalized_bid_price = 0
        if self.historical_bid_price_std != 0:
            self.normalized_bid_price = (self.bid_price - self.historical_bid_price_mean) / self.historical_bid_price_std """

        self.normalized_export_tariff = 0
        if self.historical_export_tariff_std != 0:
            self.normalized_export_tariff = (self.export_tariff - self.historical_export_tariff_mean) / self.historical_export_tariff_std
        
        self.normalized_import_tariff = 0
        if self.historical_import_tariff_std != 0:
            self.normalized_import_tariff = (self.import_tariff - self.historical_import_tariff_mean) / self.historical_import_tariff_std
        
        self.normalized_sugar_price = 0
        if self.historical_sugar_price_std != 0:
            self.normalized_sugar_price = (self.sugar_price - self.historical_sugar_price_mean) / self.historical_sugar_price_std
        
        self.normalized_sunlight = 0
        if self.historical_sunlight_std != 0:
            self.normalized_sunlight = (self.sunlight - self.historical_sunlight_mean) / self.historical_sunlight_std
        
        self.normalized_transport_fees = 0
        if self.historical_transport_fees_std != 0:
            self.normalized_transport_fees = (self.transport_fees - self.historical_transport_fees_mean) / self.historical_transport_fees_std

        #self.MVI_multiplier = (self.normalized_ask_price * 0.1) + (self.normalized_bid_price * 0.1) + (self.normalized_export_tariff * 0.1) + (self.normalized_import_tariff * 0.1) + (self.normalized_sugar_price * 0.1) + (self.normalized_sunlight * -0.4) + (self.normalized_transport_fees * 0.1)
        #self.acceptable_buy_price = current_average_mid_price * self.MVI_multiplier
        #self.acceptable_buy_price = self.historical_average_mid_price * self.MVI_multiplier

        self.export_tariff_weight = 0.1
        self.import_tariff_weight = 0.1
        self.sugar_price_weight = 0.1
        self.sunlight_weight = -0.4
        self.transport_fees_weight = 0.1

        self.MVI_multiplier = (self.normalized_export_tariff * self.export_tariff_weight) + \
                              (self.normalized_import_tariff * self.import_tariff_weight) + \
                              (self.normalized_sugar_price * self.sugar_price_weight) + \
                              (self.normalized_sunlight * self.sunlight_weight) + \
                              (self.normalized_transport_fees * self.transport_fees_weight)
        
        self.hybrid_average_mid_price = (0.3 * self.historical_average_mid_price) + (0.7 * self.current_average_mid_price)
        self.acceptable_buy_price = self.hybrid_average_mid_price * self.MVI_multiplier
        self.acceptable_sell_price = self.acceptable_buy_price

def make_empty_order_history(products):
    order_history = {}
    for product in products:
        order_history[product] = []
    
    return order_history

def make_empty_position_dictionary(products):
    position_dictionary = {}
    for product in products:
        position_dictionary[product] = 0
    
    return position_dictionary

def initialize_product_information(products, sell_order_history, buy_order_history, current_positions, observation_info_history, current_observation_info):
    product_info = {}
    for product in products:
        if product == "MAGNIFICENT_MACARONS":
            product_info["MAGNIFICENT_MACARONS"] = Macaron(product, sell_order_history[product], buy_order_history[product], current_positions[product], observation_info_history, current_observation_info)
            continue
        product_info[product] = Product(product, sell_order_history[product], buy_order_history[product], current_positions[product])
    
    # Set picnic basket buy and sell thresholds
    croissants = product_info["CROISSANTS"].average_mid_price * 6
    jams = product_info["JAMS"].average_mid_price * 3
    djembe = product_info["DJEMBES"].average_mid_price
    product_info["PICNIC_BASKET1"].set_acceptable_buy_price(croissants + jams + djembe)
    product_info["PICNIC_BASKET1"].set_acceptable_sell_price(croissants + jams + djembe)

    croissants = product_info["CROISSANTS"].average_mid_price * 4
    jams = product_info["JAMS"].average_mid_price * 2
    product_info["PICNIC_BASKET2"].set_acceptable_buy_price(croissants + jams)
    product_info["PICNIC_BASKET2"].set_acceptable_sell_price(croissants + jams)

    # Manual offset adjustments
    product_info["RAINFOREST_RESIN"].set_buy_price_offset(0)
    product_info["RAINFOREST_RESIN"].set_sell_price_offset(0)

    product_info["KELP"].set_buy_price_offset(0)
    product_info["KELP"].set_sell_price_offset(3)

    product_info["SQUID_INK"].set_buy_price_offset(-1)

    product_info["CROISSANTS"].set_buy_price_offset(-4)
    product_info["DJEMBES"].set_buy_price_offset(-4)
    product_info["JAMS"].set_buy_price_offset(-4)

    product_info["PICNIC_BASKET1"].set_buy_price_offset(-5)
    product_info["PICNIC_BASKET1"].set_sell_price_offset(product_info["PICNIC_BASKET1"].default_offset)

    product_info["PICNIC_BASKET2"].set_buy_price_offset(-5)
    product_info["PICNIC_BASKET2"].set_sell_price_offset(product_info["PICNIC_BASKET2"].default_offset)

    # Return the products' information
    return product_info

def get_position_limits():
    POSITION_LIMITS = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75
        }
    
    return POSITION_LIMITS

def get_orders(s):
    s = s.strip("{}")
    s = s.split("]")

    newList = []
    for entry in s:
        if entry != "":
            newList.append((entry + "]").strip(", "))

    d = {}
    for item in newList:
        key_value_pair = item.split(":")
        key = key_value_pair[0].strip(" '")
        
        values = key_value_pair[1].strip(" []").split(",")
        
        if values == ['']:
            d[key] = []
            
        else:
            for index, value in enumerate(values):
                values[index] = float(value.strip())
            
            d[key] = values
    
    return d

def get_positions(s):
    s = s.strip("{}")
    s = s.split(",")
    
    newList = []
    for entry in s:
        if entry != "":
            newList.append((entry).strip())

    d = {}
    for item in newList:
        key_value_pair = item.split(":")
        key = key_value_pair[0].strip("'")
        
        value = int(key_value_pair[1].strip())
        d[key] = value
    
    return d

def convert_trading_data(s):
    s = s.strip("[]")
    s = s.split("}")

    dList = []
    for entry in s:
        if entry != "":
            dList.append((entry + "}").strip(", "))
    
    sell_orders = get_orders(dList[0])
    buy_orders = get_orders(dList[1])
    positions = get_positions(dList[2])
    macaron_info = get_orders(dList[3])

    dList[0] = sell_orders
    dList[1] = buy_orders
    dList[2] = positions
    dList[3] = macaron_info
    
    return dList

def get_average(prices):
    if len(prices) == 0:
        return 0
    
    return sum(prices) / len(prices)

def voucher_makes_sense(voucher_amount, most_recent_volcanic_rock_sell_order):
    upper_bound = most_recent_volcanic_rock_sell_order * 1.02
    lower_bound = most_recent_volcanic_rock_sell_order * 0.98

    if voucher_amount < upper_bound and voucher_amount > lower_bound:
        print(f"Voucher amount {voucher_amount} DOES (YES) makes sense for most recent volcanic rock sell price {most_recent_volcanic_rock_sell_order}")
        return True
    
    print(f"Voucher amount {voucher_amount} DOES NOT (NO) make sense for most recent volcanic rock sell price {most_recent_volcanic_rock_sell_order}")
    return False

def get_lowest_sell_order(sell_orders):
    lowest_price = 0
    associated_amount = 0

    for index, sell_order in enumerate(sell_orders):
        if index == 0:
            lowest_price = sell_order[0]
            associated_amount = sell_order[1]
            continue
        
        if sell_order[0] < lowest_price:
            lowest_price = sell_order[0]
            associated_amount = sell_order[1]
    
    return (lowest_price, associated_amount)

def get_highest_buy_order(buy_orders):
    highest_price = 0
    associated_amount = 0

    for index, buy_order in enumerate(buy_orders):
        if index == 0:
            highest_price = buy_order[0]
            associated_amount = buy_order[1]
            continue
        
        if buy_order[0] > highest_price:
            highest_price = buy_order[0]
            associated_amount = buy_order[1]
    
    return (highest_price, associated_amount)

def buy_to_bot(orders, current_position, position_limit, product, best_ask, best_ask_amount):
    if current_position - best_ask_amount <= position_limit:
        orders.append(Order(product, best_ask, -1 * best_ask_amount))

def sell_to_bot(orders, current_position, position_limit, product, best_bid, best_bid_amount):
    if current_position - best_bid_amount >= (-1 * position_limit):
        orders.append(Order(product, best_bid, -1 * best_bid_amount))

def big_dip_checker(order_history, current_order, multiplier):
    historical_average = get_average(order_history) 
    return current_order > (historical_average * multiplier)

def small_dip_checker(order_history, recents_length, current_order, multiplier):
    recents = order_history
    if len(recents) > recents_length:
        recents = recents[0:recents_length]
    
    recents_average = get_average(recents) 
    #print(f"recents_average: {recents_average}")

    return current_order > (recents_average * multiplier)

def update_sell_order_history(history, product, new_addition):
    max_history_length = 150

    if len(history[product]) > max_history_length:
        history[product].pop(0)
    history[product].append(new_addition)

def update_buy_order_history(history, product, new_addition):
    max_history_length = 150

    if len(history[product]) > max_history_length:
        history[product].pop(0)
    history[product].append(new_addition)

class Trader:
    def run(self, state: TradingState):
        PRODUCT_NAMES = ["RAINFOREST_RESIN",
                         "KELP",
                         "SQUID_INK",
                         "CROISSANTS",
                         "DJEMBES",
                         "JAMS",
                         "PICNIC_BASKET1",
                         "PICNIC_BASKET2",
                         "VOLCANIC_ROCK",
                         "VOLCANIC_ROCK_VOUCHER_9500",
                         "VOLCANIC_ROCK_VOUCHER_9750",
                         "VOLCANIC_ROCK_VOUCHER_10000",
                         "VOLCANIC_ROCK_VOUCHER_10250",
                         "VOLCANIC_ROCK_VOUCHER_10500",
                         "MAGNIFICENT_MACARONS"]

        POSITION_LIMITS = get_position_limits()
        
        MACARON_INFO = ["askPrice",
                        "bidPrice",
                        "exportTariff",
                        "importTariff",
                        "sugarPrice",
                        "sunlightIndex",
                        "transportFees"]
        
        # Print state properties
        #print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        #print(f"Own trades: {state.own_trades}")

        # Make relavant dictionaries (by default)
        sell_order_history = make_empty_order_history(PRODUCT_NAMES)
        buy_order_history = make_empty_order_history(PRODUCT_NAMES)
        current_positions = make_empty_position_dictionary(PRODUCT_NAMES)
        previous_macaron_information = make_empty_order_history(MACARON_INFO)

        # Update the dictionaries with previous trading data if it exists
        if state.traderData != "":
            sell_order_history, buy_order_history, current_positions, previous_macaron_information = convert_trading_data(state.traderData)
        
        macaron_state = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
        print(f"WE'RE BREATHING SUNLIGHTTT {macaron_state.sunlightIndex}")

        products = initialize_product_information(PRODUCT_NAMES, sell_order_history, buy_order_history, current_positions, previous_macaron_information, macaron_state)
        
        previous_macaron_information["askPrice"].append(macaron_state.askPrice)
        previous_macaron_information["bidPrice"].append(macaron_state.bidPrice)
        previous_macaron_information["exportTariff"].append(macaron_state.exportTariff)
        previous_macaron_information["importTariff"].append(macaron_state.importTariff)
        previous_macaron_information["sugarPrice"].append(macaron_state.sugarPrice)
        previous_macaron_information["sunlightIndex"].append(macaron_state.sunlightIndex)
        previous_macaron_information["transportFees"].append(macaron_state.transportFees)

        # Orders to be placed on exchange matching engine
        result = {}

        # state.order_depths:
        # keys = products, values = OrderDepth instances

        # Go through each product, for each product
        for product in state.order_depths:
            print(f"Current product: {product}")

            """
            OrderDepth contains the collection of all outstanding buy and sell orders
            (or “quotes”) that were sent by the trading bots for a certain symbol

            buy_orders and sell_orders dictionaries:
            Key = price associated with the order
            Value = total volume on that price level
            """
            order_depth: OrderDepth = state.order_depths[product]

            # Skip the first iteration of trading, also tariffs are scary (boo) (oh no) (spooky)
            #if state.traderData == "" or product == "MAGNIFICENT_MACARONS":
            if state.traderData == "":
                #print("First iteration, will not do any trading")
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = get_lowest_sell_order(list(order_depth.sell_orders.items()))
                    update_sell_order_history(sell_order_history, product, best_ask)

                    best_bid, best_bid_amount = get_highest_buy_order(list(order_depth.buy_orders.items()))
                    update_buy_order_history(buy_order_history, product, best_bid)
                continue
            
            # Get the current position of the product
            position = products[product].position
            #print(f"Current position: {position}")
            
            # Initialize the product class
            #Rainforest_Resin = Product("RAINFOREST_RESIN", sell_order_history["RAINFOREST_RESIN", ])

            # Make a list of orders
            orders: List[Order] = []
            
            # Get the thresholds to buy and sell for this specific product
            acceptable_buy_price = products[product].acceptable_buy_price
            acceptable_sell_price = products[product].acceptable_sell_price

            print(f"Acceptable buy price: {acceptable_buy_price}")
            print(f"Acceptable sell price: {acceptable_sell_price}")

            # I guess... how many buy and sell orders?
            #print(f"Buy Order depth: {len(order_depth.buy_orders)}, Sell order depth: {len(order_depth.sell_orders)}")

            # Make conditions (for a crash or not) in which we would want to sell everything
            best_ask, best_ask_amount = get_lowest_sell_order(list(order_depth.sell_orders.items()))

            # Condition 1: Sell order is a too high above the historical average (big-dip checker)
            # Condition 2: Sell order is a slightly higher than a recent average (small-dip checker)
            # Condition 3: Sell order of PICNIC_BASKET1 and PICNIC_BASKET2 is a slightly higher than a recent average (small-dip checker)
            # Condition 4: Sell order of DJEMBES is a slightly higher than a recent average (small-dip checker)
            # Either needs to be true for us to sell everything
            condition_one = big_dip_checker(products[product].sell_order_history, best_ask, 1.08)
            condition_two = small_dip_checker(products[product].sell_order_history, 20, best_ask, 1.04)
            condition_three = small_dip_checker(products[product].sell_order_history, 10, best_ask, 1.08) and (product in ["PICNIC_BASKET1", "PICNIC_BASKET2"])
            condition_four = small_dip_checker(products[product].sell_order_history, 40, best_ask, 1.08) and (product == "DJEMBES")

            if (condition_one or condition_two or condition_three or condition_four):
                print("I'M GOING TO CRASH OUT !!!")
                update_sell_order_history(sell_order_history, product, best_ask)

                best_bid, best_bid_amount = get_highest_buy_order(list(order_depth.buy_orders.items()))
                update_buy_order_history(buy_order_history, product, best_bid)

                # Sell everything (sell to all buy orders until position <= 0)
                for buy_order in list(order_depth.buy_orders.items()):
                    bid, bid_amount = buy_order
                    if position <= 0:
                        break
                    
                    # In case I guess (ideally I would think that we sell everything until our position is back to 0)
                    if position - bid_amount <= 0:
                        orders.append(Order(product, bid, -1 * position))

                    else:
                        orders.append(Order(product, bid, -1 * bid_amount))
                    
                    position -= bid_amount
                
                # We're done for this product; move onto the next product
                continue
            
            # Otherwise, if there is no reason to sell everything, resume!

            # If there are sell orders that exist (if bots are selling)
            if len(order_depth.sell_orders) != 0:
                # Get the price and quantity of the first sell?
                # best_ask = price
                # best_ask_amount = quantity
                best_ask, best_ask_amount = get_lowest_sell_order(list(order_depth.sell_orders.items()))
                #print(f"Sell orders: {list(order_depth.sell_orders.items())}")

                # Add the lowest sell order to sell_order_history
                # Default: Keep the past 150 orders
                """ if len(sell_order_history[product]) > 150:
                    sell_order_history[product].pop(0)
                sell_order_history[product].append(best_ask) """
                update_sell_order_history(sell_order_history, product, best_ask)

                # or best_ask < (sell_order_history[product][-5] * 0.92)

                # Previously: Big jumps: Keep the past 35 orders
                # Big jumps: Keep the past 80 orders
                # TODO: Maybe make the multiplier 1.001?

                # If there are more than 75 sell orders in the sell order history (max is 150)
                # TODO: Discuss about this
                if len(sell_order_history[product]) > 7500:
                    # If the current sell order price is 0.5% above the 10th most recent sell order price
                    if best_ask > (sell_order_history[product][-10] * 1.05):
                        print("BIG JUMP!!!")

                        # Reduce the sell order history by 70 (remove the 70 oldest)
                        for i in range(70, len(sell_order_history[product])):
                            sell_order_history[product].pop(0)

                sell_order_history[product].append(best_ask)
                
                # If the bot is selling for less than we expect (wahoo)
                if int(best_ask) < acceptable_buy_price:
                    # Buy some of that I guess
                    print(f"BUY {(-1 * best_ask_amount)} x {best_ask}")
                    buy_to_bot(orders, position, POSITION_LIMITS[product], product, best_ask, best_ask_amount)
                    #orders.append(Order(product, best_ask, -1 * best_ask_amount))
                    position += best_ask_amount

            # If there are buy orders that exist (if bots are buying)
            if len(order_depth.buy_orders) != 0:
                # Get the price and quantity of the first buy?
                # best_bid = price
                # best_bid_amount = quantity
                best_bid, best_bid_amount = get_highest_buy_order(list(order_depth.buy_orders.items()))
                print(f"Buy orders: {list(order_depth.buy_orders.items())}")
                
                update_buy_order_history(buy_order_history, product, best_bid)

                # If the bot is buying for more than we expect (wahoo)
                if int(best_bid) > acceptable_sell_price:
                    # Sell some of that I guess
                    print(f"SELL {best_bid_amount} x {best_bid}")
                    #sell_to_bot(orders, position, POSITION_LIMITS[product], product, best_bid, best_bid_amount)
                    orders.append(Order(product, best_bid, -1 * best_bid_amount))
                    position -= best_bid_amount
                
            # This is still in the "for product in state.order_depths" for loop
            # After we make our orders, put those orders in result for that respective product
            result[product] = orders
            current_positions[product] = position

        #print(f"WHAT IS SELL: {sell_order_history}")

        newData = []
        newData.append(sell_order_history)
        newData.append(buy_order_history)
        newData.append(current_positions)
        newData.append(previous_macaron_information)

        # String value holding Trader state data required. 
        # It will be delivered as TradingState.traderData on next execution.
        traderData = str(newData)

        # Sample conversion request. Check more details below. 
        conversions = 0
        return result, conversions, traderData