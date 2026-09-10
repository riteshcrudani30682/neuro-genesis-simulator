"""Immediate reward is separate from lifetime evolutionary fitness."""


def reward_components(*, food=False, hazard=False, blocked=False, died=False):
    return {'survival': 0.01 if not died else 0.0, 'food': 2.0 if food else 0.0,
            'hazard': -2.0 if hazard else 0.0, 'blocked': -0.05 if blocked else 0.0,
            'death': -1.0 if died else 0.0}
