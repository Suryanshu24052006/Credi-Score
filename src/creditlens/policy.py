# decision thresholds - fixed business settings, chosen before the test set was scored.
# reports/cutoffs_test.csv shows the approval / default-rate trade-off if a bank wants to move them
APPROVE_MAX_PD = 0.05
DECLINE_MIN_PD = 0.15

GRADES = [(0.02, "A"), (0.05, "B"), (0.10, "C"), (0.20, "D"), (1.01, "E")]


def decide(pd_value):
    if pd_value < APPROVE_MAX_PD:
        return "approve"
    if pd_value >= DECLINE_MIN_PD:
        return "decline"
    return "manual_review"


def grade(pd_value):
    for upper, g in GRADES:
        if pd_value < upper:
            return g
    return "E"
