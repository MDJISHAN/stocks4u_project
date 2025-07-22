# swp_calculator.py

def swp_calculator(initial_investment, withdrawal_amount, annual_return, months):
    balance = initial_investment
    monthly_return = (1 + (annual_return / 100)) ** (1 / 12) - 1
    swp_schedule = []

    for month in range(1, months + 1):
        balance = balance * (1 + monthly_return)
        balance -= withdrawal_amount

        if balance < 0:
            swp_schedule.append({"Month": month, "Balance": 0})
            break

        swp_schedule.append({"Month": month, "Balance": round(balance, 2)})

    return swp_schedule
