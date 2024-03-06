
def calc_premium(close_price, index_price, expire_day_count):
    premium = close_price - index_price
    premium_percentage = premium / index_price * 100
    annualized_premium = premium_percentage * 365 / expire_day_count
    # print it.
    print("premium: ", premium, "premium_percentage: ", premium_percentage, "annualized_premium: ", annualized_premium)
    return premium, premium_percentage, annualized_premium

def calc_premium_unit_test():
    close_price = 5101
    index_price = 5342
    expire_day_count = 198
    calc_premium(close_price, index_price, expire_day_count)

if __name__ == "__main__":
    calc_premium_unit_test()
    print("Done.")