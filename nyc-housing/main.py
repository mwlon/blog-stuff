import numpy as np
from dataclasses import dataclass
import dataclasses
import pandas as pd

# Everything is annual!
# A lot of this is NYC-specific!
# I'm not accounting for things like the federal standard
# deduction changing over time... that's too complicated.

@dataclass
class Scenario:
  name: str
  is_coop: bool
  price: float
  maintenance: float

  years: int = 20 # between buying and selling
  taxes: float = 0.0
  down_payment_ratio: float = 0.2
  mortgage_years: int = 30
  mortgage_interest_rate: float = 0.06
  # with houseit or something the rebate might be up to 0.02 (2%)
  broker_rebate: float = 0.0
  utilities_if_condo: float = 600

  stock_return: float = 1.068
  real_estate_return: float = 1.055
  federal_standard_deduction: float = 13850
  state_standard_deduction: float = 8025
  other_itemized_deduction: float = 5400
  # marginal tax rates
  federal_tax_rate: float = 0.37
  state_tax_rate: float = 0.0685 + 0.03876
  capital_gains_rate: float = 0.2

  def __str__(self):
    return self.name

# all the types of cash flows
categories = [
  'maintenance',
  'closing',
  'principal',
  'interest',
  'taxes',
  'tax_savings',
]

@dataclass
class Result:
  cash_flows: np.ndarray # categories x (years + 1)
  s: Scenario

  def show(self):
    # to compare vs rent, assume rent rises at the
    # same rate as real estate price
    annual_discount = 1 / self.s.stock_return
    rent_growth = self.s.real_estate_return * annual_discount
    years = self.s.years
    if rent_growth - 1 == 0:
      w = years
    else:
      w = (1 - rent_growth ** years) / (1 - rent_growth)

    discounting = annual_discount ** np.arange(years + 1)
    discounted_cash_flows = self.cash_flows * discounting[None, :]
    nonrec_by_category = np.sum(discounted_cash_flows, axis=1)
    nonrec = np.sum(nonrec_by_category)
    print(f'effective annual rent: ${nonrec / w:.0f}')
    for category, amount in zip(categories, nonrec_by_category):
      pct = 100 * amount / nonrec
      print(f'\t{category}: {pct:.2f}%')

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 300)
    pd.set_option('display.float_format', '{:.0f}'.format)

    print('cash flows by year (raw):')
    df = pd.DataFrame(self.cash_flows.transpose(), columns=categories)
    df['all'] = np.sum(self.cash_flows, axis=0)
    print(df)

    print('cash flows by year (discounted):')
    df = pd.DataFrame(discounted_cash_flows.transpose(), columns=categories)
    df['all'] = np.sum(discounted_cash_flows, axis=0)
    print(df)

def calc_closing(s):
  buy_cost = 0
  sell_cost = 0

  # condo vs coop closing
  if s.is_coop:
    # misc
    buy_cost += 5000
    sell_cost += 5000

    # flip tax (sell)
    sell_cost += 0.015 * s.price
  else:
    # misc
    buy_cost += 10000
    sell_cost += 10000

    # mortgage recording tax (buy)
    mortgage_amount = s.price * (1 - s.down_payment_ratio)
    buy_cost += 0.01925 * mortgage_amount

  # brokers (sell)
  sell_cost += 0.06 * s.price
  buy_cost -= s.broker_rebate * s.price

  # transfer fee (usually sell)
  sell_cost += 0.01825 * s.price

  # mansion tax (buy) (yes it really isn't bracketed)
  if s.price < 1000000:
    mansion_tax_rate = 0.0
  elif s.price < 2000000:
    mansion_tax_rate = 0.01
  elif s.price < 3000000:
    mansion_tax_rate = 0.0125
  else:
    raise ValueError('unimplemented mansion tax level')
  buy_cost += mansion_tax_rate * s.price

  res = np.zeros(s.years + 1)
  res[0] = buy_cost
  res[-1] = sell_cost * s.real_estate_return ** s.years
  return res

def calc(s):
  # one extra year for the cashflow from selling the place
  res = np.zeros([len(categories), s.years + 1])

  # 0: MAINTENANCE
  base_maintenance = s.maintenance
  if not s.is_coop:
    base_maintenance += s.utilities_if_condo

  maintenance = base_maintenance * (s.real_estate_return ** np.arange(s.years))
  res[0, :s.years] = maintenance

  # 1: CLOSING
  res[1] = calc_closing(s)

  # 2: PRINCIPAL
  # typical payment schedule is fixed payments;
  # initially more interest, dropping to more principal toward the end
  loan = s.price * (1 - s.down_payment_ratio)
  r = 1 + s.mortgage_interest_rate
  fixed_payment = loan * s.mortgage_interest_rate * r ** s.mortgage_years / (r ** s.mortgage_years - 1)
  n = min(s.years, s.mortgage_years)
  principal_payment = fixed_payment * r ** np.arange(n) - loan * r ** np.arange(n) * (r - 1)
  res[2, :n] = principal_payment
  res[2, 0] += s.price * s.down_payment_ratio

  debt = np.zeros([s.years + 1])
  debt[0] = loan
  debt[1:n + 1] = loan - np.cumsum(principal_payment)
  final_debt = debt[-1]
  # final money recovered by selling home (after paying off the rest of debt to bank)
  sell_price = s.price * s.real_estate_return ** s.years
  sell_profit = sell_price - s.price
  sell_taxes = max(0, sell_profit - 500000) * s.capital_gains_rate
  res[2, -1] = final_debt + sell_taxes - sell_price

  # 3: INTEREST
  res[3, :n] = fixed_payment - principal_payment

  # 4: TAXES
  if s.is_coop and s.taxes > 0:
      raise ValueError('coops dont have taxes')
  taxes = s.taxes * (s.real_estate_return ** np.arange(s.years))
  res[4, :s.years] = taxes

  # 5: TAX SAVINGS
  deductible_mortgage_interest = s.mortgage_interest_rate * np.minimum(debt[:s.years], 750000)
  itemized_deduction = s.other_itemized_deduction + deductible_mortgage_interest + np.minimum(taxes, 10000)
  federal_incremental_deduction = np.maximum(itemized_deduction - s.federal_standard_deduction, 0)
  state_incremental_deduction = np.maximum(itemized_deduction - s.state_standard_deduction, 0)
  income_tax_savings = s.federal_tax_rate * federal_incremental_deduction + s.state_tax_rate * state_incremental_deduction
  res[5, :s.years] = -income_tax_savings

  return Result(res, s)
    

scenarios = [
  Scenario(
    "2M mortgage",
    is_coop=False,
    price=2000000 - 1,
    taxes=20000,
    maintenance=18000,
    years=20,
  ),
  Scenario(
    "2M cash",
    is_coop=False,
    price=2000000 - 1,
    taxes=20000,
    maintenance=18000,
    down_payment_ratio=1.0,
    years=20,
  ),
]
 
for s in scenarios:
  print(f"SCENARIO: {s}\n=====================")
  calc(s).show()
  print("")
