# sip_calculator.py

def sip_calculator(amount, yearly_rate, years):
    monthly_rate = yearly_rate / 12 / 100
    months = years * 12
    future_value = amount * ((((1 + monthly_rate) ** months) - 1) * (1 + monthly_rate)) / monthly_rate
    return round(future_value)
